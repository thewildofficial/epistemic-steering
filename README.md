# Epistemic Steering

Probe-based LLM steering for hallucination prevention.

## Overview

This project develops methods to detect and prevent hallucinations in large language models by using epistemic confidence probes trained on hidden states. When confidence is low, steering interventions modify model behavior to avoid unreliable outputs.

## Setup

```bash
uv sync
```

## Structure

```
epistemic-steering/
├── src/              # Core modules (probe, steering, evaluate, data)
├── scripts/          # Training and evaluation scripts
├── notebooks/         # Jupyter notebooks for analysis
├── tests/            # Unit and integration tests
├── paper/            # Paper figures and tables
├── data/             # Dataset storage (gitignored)
├── figures/          # Generated figures (gitignored)
└── pyproject.toml    # UV project configuration
```

## Key Components

- **probe.py**: Confidence scoring from hidden states
- **steering.py**: Activation steering interventions
- **evaluate.py**: Hallucination and accuracy metrics
- **data.py**: MMLU, GSM8K dataset loading