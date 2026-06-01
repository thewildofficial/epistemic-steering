# Epistemic Steering

**Using hidden-state probes to route LLM behavior and prevent hallucinations.**

When an LLM outputs a confident answer to a question it cannot reliably answer, that output is functionally a lie. This project builds a steering system that surfaces mathematically grounded uncertainty from hidden states and routes the model accordingly—direct answer, chain-of-thought reasoning, or abstention.

**Author:** Aban Hasan, BITS Pilani (2025eb01715@online.bits-pilani.ac.in)  
**Paper:** [arXiv preprint](paper/paper.pdf) (11 pages)

---

## Key Results

| | In-Sample | Held-Out (56 subjects) |
|---|---|---|
| **MMLU AUROC** | 0.827 | **0.957** |
| **Direct Accuracy** | — | **89.5%** |
| **Prevention Rate** | 78.2% (t=0.50) | — |
| **GSM8K AUROC** | 0.967 | — |

**Core finding:** LLMs encode uncertainty about *both* factual and reasoning tasks in their hidden states. The apparent "knowing-that / knowing-how asymmetry" was a **measurement artifact** — the original GSM8K setup achieved only 3.5% model accuracy from inadequate prompting. With 4-shot CoT + thinking mode, accuracy rose to 87.3% and probe AUROC reached 0.967. The probe detects correctness universally across task types.

## Architecture

```
QUESTION → PREFILL PROBE → confidence score
    │
    ├── ≥ 0.7 → DIRECT ANSWER
    ├── 0.3-0.7 → CHAIN-OF-THOUGHT + GENERATION-TIME MONITORING
    └── ≤ 0.3 → ABSTAIN ("I don't know")
```

- **Model:** Qwen3.5-4B-Instruct (32 layers, 2560 hidden dim)
- **Probe:** Logistic regression with analytically derived weights (no gradient updates to base model)
- **Layer:** **25**, last-token prefill activation
- **Calibration:** Platt scaling (ECE −32.1%)
- **Router:** PrefillProbeRouter with configurable thresholds

## Setup

```bash
git clone https://github.com/thewildofficial/epistemic-steering.git
cd epistemic-steering
uv sync
uv run pytest tests/  # 146 tests
```

## Structure

```
├── src/              # probe.py, steering.py, evaluate.py, data.py, plotting.py
├── scripts/          # verify, threshold analysis, generate figures, evaluate held-out
├── notebooks/        # Jupyter notebooks for exploration
├── tests/            # 146 tests (pytest)
├── paper/            # LaTeX source + compiled PDF
├── data/             # Experiment data (gitignored, stored on Modal volume)
├── figures/          # Generated plots (PNG + PDF)
└── pyproject.toml    # UV project configuration
```

## Citation

```bibtex
@misc{hasan2026epistemic,
  title={Epistemic Steering: Using Hidden-State Probes to Route LLM Behavior and Prevent Hallucinations},
  author={Hasan, Aban},
  year={2026},
  note={arXiv preprint}
}
```

## License

MIT
