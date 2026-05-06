"""Dataset loading and preprocessing.

Handles MMLU, GSM8K, and custom evaluation datasets.
Manages data splits, tokenization, and batch formation.

Core functions:
- load_mmlu: Load MMLU benchmark dataset
- load_gsm8k: Load GSM8K math reasoning dataset
- preprocess_dataset: Tokenize and format inputs
- create_splits: Train/val/test partitioning
"""