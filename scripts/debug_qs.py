"""Debug: inspect individual question file format."""
import pickle, os
import modal
from modal import App, Image, Volume

app = App("debug-qs")
volume = Volume.from_name("epistemic-model-cache")
GEN_TIME_DIR = "/vol/results/gen_time_gsm8k_layer25_qwen_prompt"
QUESTIONS_DIR = f"{GEN_TIME_DIR}/questions"
ALL_PKL = f"{GEN_TIME_DIR}/all_results.pkl"

image = Image.debian_slim().pip_install("numpy", "tqdm")

@app.function(image=image, volumes={"/vol": volume}, cpu=2.0, timeout=600)
def debug():
    import numpy as np
    
    # Load all_results.pkl first
    with open(ALL_PKL, 'rb') as f:
        all_results = pickle.load(f)
    if isinstance(all_results, list):
        all_results = {r.get('question_id', str(i)): r for i, r in enumerate(all_results)}
    
    n_with_hs = sum(1 for r in all_results.values() if isinstance(r, dict) and 'hidden_states' in r and len(r.get('hidden_states', [])) > 0)
    print(f"all_results.pkl: {len(all_results)} total, {n_with_hs} with hidden_states")
    
    # Show a few entries without hidden_states
    no_hs = [qid for qid, r in all_results.items() if isinstance(r, dict) and ('hidden_states' not in r or len(r.get('hidden_states',[])) == 0)]
    print(f"Entries without HS: {len(no_hs)}")
    if no_hs:
        sample_qid = no_hs[0]
        sample = all_results[sample_qid]
        print(f"  Sample {sample_qid}: keys={list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
    
    # Now try to load one individual file
    qfiles = sorted([f for f in os.listdir(QUESTIONS_DIR) if f.endswith('.pkl')])
    print(f"\nIndividual files: {len(qfiles)}")
    
    for qf in qfiles[:3]:
        qid = qf.replace('.pkl', '')
        with open(f"{QUESTIONS_DIR}/{qf}", 'rb') as f:
            data = pickle.load(f)
        
        in_main = qid in all_results
        in_main_hs = in_main and isinstance(all_results[qid], dict) and 'hidden_states' in all_results[qid]
        
        if isinstance(data, dict):
            has_hs = 'hidden_states' in data
            if has_hs:
                hs = data['hidden_states']
                print(f"\n{qid}: in_main={in_main}, in_main_has_hs={in_main_hs}")
                print(f"  hs type: {type(hs).__name__}")
                if isinstance(hs, np.ndarray):
                    print(f"  hs shape: {hs.shape}, dtype: {hs.dtype}")
                    print(f"  len(hs): {len(hs)}")
                elif isinstance(hs, list):
                    print(f"  hs list len: {len(hs)}")
                    if hs:
                        print(f"  hs[0] type: {type(hs[0]).__name__}")
                print(f"  correct: {data.get('correct')}")
            else:
                print(f"\n{qid}: has_hs=False, keys={list(data.keys())[:8]}")
        else:
            print(f"\n{qid}: not dict, type={type(data).__name__}")

@app.local_entrypoint()
def main():
    debug.remote()
