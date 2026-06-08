#!/usr/bin/env python3
"""Diagnostic script for Hewitt-Liang controls data."""

import json, sys, os, numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def diagnose_benchmark(benchmark):
    print(f"\n{'='*60}\nBENCHMARK: {benchmark}\n{'='*60}")
    # Try v2 first
    v2_dir = PROJECT_ROOT / "data" / "benchmark_activations_v2" / benchmark
    if v2_dir.is_dir():
        print(f"  Using V2 directory: {v2_dir}")
        json_files = sorted(v2_dir.glob("*.json"))
        labels = []
        for jf in json_files:
            with open(jf) as f:
                meta = json.load(f)
            correct = meta.get("correct", None)
            t = type(correct).__name__
            if isinstance(correct, (list, tuple)):
                t += f" len={len(correct)}"
            labels.append((jf.name, correct, t))
        print(f"  JSON files found: {len(json_files)}")
        correct_vals = []
        for name, c, t in labels:
            if isinstance(c, (list, tuple)):
                # Convert list to boolean
                c_bool = bool(c)  # This might be wrong!
            elif isinstance(c, np.ndarray):
                c_bool = c.item() if c.ndim == 0 else bool(c)
            else:
                c_bool = bool(c) if c is not None else None
            correct_vals.append(c_bool)
        print(f"  Label distribution (bool conversion): {Counter(correct_vals)}")
        list_types = [t for _, _, t in labels if "list" in t or "tuple" in t or "ndarray" in t]
        if list_types:
            print(f"  Non-scalar correct types found: {list_types[:5]}")

    # Try legacy
    legacy_dir = PROJECT_ROOT / "data" / f"benchmark_activations_{benchmark}"
    legacy_jsonl = PROJECT_ROOT / "data" / f"benchmark_activations_{benchmark}_results.jsonl"
    if legacy_dir.is_dir() and legacy_jsonl.is_file():
        print(f"  Legacy JSONL found: {legacy_jsonl}")
        with open(legacy_jsonl) as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"  JSONL records: {len(records)}")
        correct_vals = [bool(r.get("correct", False)) for r in records]
        print(f"  Label distribution: {Counter(correct_vals)}")

    # Also load npy to check for duplicates
    if v2_dir.is_dir():
        npy_dir = v2_dir
    elif legacy_dir.is_dir():
        sub = legacy_dir / benchmark
        if sub.is_dir():
            npy_dir = sub
        else:
            npy_dir = legacy_dir
    else:
        return
    npy_files = sorted(npy_dir.glob("*.npy"))
    print(f"  NPY files: {len(npy_files)}")
    if npy_files:
        shapes = [tuple(np.load(f).shape) for f in npy_files[:5]]
        print(f"  Sample shapes: {shapes}")
        # Check for exact duplicate files
        hashes = []
        for f in npy_files[:min(len(npy_files), 100)]:
            arr = np.load(f)
            hashes.append(hash(arr.tobytes()))
        dupes = len(hashes) - len(set(hashes))
        print(f"  Exact duplicates among first 100: {dupes}")

for b in ["mmlu", "arc_challenge", "triviaqa", "gsm8k", "math", "humaneval"]:
    diagnose_benchmark(b)
print("\nDone.")
