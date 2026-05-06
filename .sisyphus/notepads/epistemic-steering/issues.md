
## F3 Manual QA — 2026-05-06

**Code Imports FAIL**: `steering.py:11` uses `from probe import compute_confidence` (bare import) instead of `from src.probe import compute_confidence`. User's exact import check command fails from project root. Fix: use package-relative import or ensure `src/` is on sys.path in `__init__.py`.

**All other checks passed**:
- Tests: 146 passed, 1 skipped
- Verification: MMLU AUROC 0.8274
- Steering demo: routing output correct
- Paper: compiles to 12-page PDF
- Figures: 24 files (12 PNG + 12 PDF)

## F4 Scope Fidelity Check — 2026-05-06

### Check 1: No keyword classification — CLEAN
- One incidental match in `scripts/threshold_analysis.py:333` ("easy to classify" in descriptive text), not routing logic.

### Check 2: No system prompts for steering — CLEAN
- Zero matches for `system prompt`, `role: system`, or prompt-based steering across all `.py` files.

### Check 3: No retrieval/clarification/calculator actions — CLEAN
- Zero matches for retrieve, clarify, calculator, RAG, or search actions across all `.py` files.

### Check 4: Budget check — CLEAN (with caveat)
- Held-out eval actual costs tracked in JSON: ~$0.23 total across 3 summary files.
- Other Modal scripts have cost tracking and ceilings well below $28 (T8: $8, T12: $10, baselines: <$1 each).
- Caveat: No centralized Modal cost log; only heldout eval summaries save `estimated_cost_usd` locally. Other runs track cost in-script but do not persist to local JSON.

### Check 5: Changes to src/probe.py, src/data.py, src/evaluate.py within spec — CLEAN
- `probe.py`: exactly 8 functions as specified (load_probe_weights, compute_confidence, score_dataset, compute_auroc, compute_confusion_matrix, compute_prevention_rate, compute_unnecessary_block_rate, threshold_sweep).
- `data.py`: exactly 6 functions as specified (load_probe_results, load_allpos_results, load_activations, validate_data, split_by_dataset, clean_mmlu_answers).
- `evaluate.py`: exactly 7 functions as specified (confusion_matrix_at_threshold, prevention_rate_at_threshold, unnecessary_block_rate_at_threshold, selective_accuracy, token_efficiency, calibration_curve, compute_all_metrics).

### Check 6: No unaccounted files — 1 MAJOR issue
- **MAJOR**: `scripts/regenerate_gsm8k_chat.py` — NOT in plan. Attempts to fix GSM8K accuracy via chat-template prompting, contradicting plan guardrail: "NO claims about reasoning steering based on 0.64 AUROC — acknowledge limitation" and "Plan scopes steering to factual (MMLU) primary, reasoning (GSM8K) as limitation analysis."
  - Script was executed: `data/gsm8k_chat/` exists with regenerated results.
  - Output leaked into paper: original data shows 3.5% GSM8K accuracy (7/200), but paper reports 35.0% (70/200) from chat-regenerated baseline.
- Minor unaccounted: `tests/test_plotting.py`, `tests/test_heldout_eval.py` (extra tests not in plan).
- Metadata: `.DS_Store` and `__pycache__` present despite `.gitignore` exclusion.

### Check 7: Module docstrings match implementations — CLEAN
- All 5 src modules (probe.py, data.py, evaluate.py, steering.py, plotting.py) have docstrings that accurately describe their functions, parameters, and return values.

### Missing files (noted for reference, not part of unaccounted check)
- `scripts/download_data.py` (T2)
- `notebooks/01_insamp_verification.ipynb` (T5)
- `notebooks/02_threshold_analysis.ipynb` (T6)
- `notebooks/03_gen_time_exploration.ipynb` (T8)
- `notebooks/04_comparison.ipynb` (T13)

### Verdict
**REJECT** due to scope contamination: `regenerate_gsm8k_chat.py` is unaccounted scope creep that regenerated GSM8K baselines, changing the paper's reported accuracy from 3.5% to 35% and violating the plan's explicit guardrail to treat GSM8K as a limitation rather than improving it.
