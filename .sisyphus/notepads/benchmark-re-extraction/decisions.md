# Decisions — Benchmark Re-Extraction

## 2026-05-19

- **GPU**: L4 at $0.80/hr (optimal price/performance within budget)
- **Budget cap**: $8.00 of $9.44 (leaving $1.44 buffer)
- **Phased**: Pilot MATH + ARC first, validate, then HumanEval + TriviaQA
- **N_QUESTIONS**: 100 per benchmark (up from 50)
- **MAX_NEW_TOKENS**: MATH=2048, HumanEval=1024, ARC=1024, TriviaQA=512
- **Chat template**: Required on all prompts via `apply_chat_template()`
- **Thinking mode**: `<tool_call>/insert>\n` suffix on all prompts
- **Few-shot**: MATH=4-shot CoT, ARC=4-shot CoT, TriviaQA=zero-shot, HumanEval=function completion
- **Probe architecture unchanged**: Layer 25, hidden state only, LogisticRegressionCV, Platt scaling