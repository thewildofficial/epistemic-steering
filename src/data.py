"""Dataset loading and validation for epistemic steering experiments.

Handles loading of probe results, activations, and data validation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_probe_results(path: str) -> pd.DataFrame:
    """Load probe results from a JSONL file.

    Args:
        path: Path to the JSONL file containing probe results.

    Returns:
        DataFrame with columns: question_id (int), dataset (str),
        prompt (str), generated_text (str), model_answer (str),
        correct (bool), top_token_prob (float).
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["question_id"] = df["question_id"].astype(int)
    df["correct"] = df["correct"].astype(bool)

    return df


def load_allpos_results(path: str) -> pd.DataFrame:
    """Load all-positions probe results from a JSONL file.

    Args:
        path: Path to the JSONL file containing all-positions probe results.

    Returns:
        DataFrame with columns: question_id (int), dataset (str),
        prompt (str), generated_text (str), model_answer (str),
        correct (bool), top_token_prob (float), token_positions (list[int]).
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["question_id"] = df["question_id"].astype(int)
    df["correct"] = df["correct"].astype(bool)

    return df


def load_activations(path: str, question_id: int, layer: int) -> np.ndarray:
    """Load activations for a specific question and layer.

    Args:
        path: Directory containing activation .npy files.
        question_id: The question ID.
        layer: The layer number (0-indexed).

    Returns:
        Numpy array of activations with shape (sequence_length, hidden_dim).

    Raises:
        FileNotFoundError: If the activation file doesn't exist.
    """
    file_path = Path(path) / f"q{question_id}_layer{layer}.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"Activation file not found: {file_path}")

    return np.load(file_path)


def validate_data(df: pd.DataFrame) -> dict:
    """Validate probe results DataFrame.

    Checks for:
    - No null values in required columns
    - correct column is boolean type
    - dataset values are 'mmlu' or 'gsm8k'

    Args:
        df: DataFrame to validate.

    Returns:
        Dict with keys:
            - missing_counts: dict mapping column name to count of null values
            - dataset_sizes: dict mapping dataset name to row count
            - errors: list of validation error strings
    """
    errors = []
    required_columns = ["question_id", "dataset", "prompt", "generated_text",
                         "model_answer", "correct", "top_token_prob"]

    missing_counts = {}
    for col in required_columns:
        if col in df.columns:
            missing_counts[col] = df[col].isnull().sum()
            if missing_counts[col] > 0:
                errors.append(f"Missing values in column: {col}")
        else:
            missing_counts[col] = len(df)
            errors.append(f"Missing required column: {col}")

    if "correct" in df.columns:
        if not df["correct"].dtype == bool:
            errors.append(f"correct column must be bool, got {df['correct'].dtype}")

    valid_datasets = {"mmlu", "gsm8k"}
    if "dataset" in df.columns:
        invalid_datasets = ~df["dataset"].isin(valid_datasets)
        if invalid_datasets.any():
            invalid_values = df.loc[invalid_datasets, "dataset"].unique()
            errors.append(f"Invalid dataset values: {invalid_values}")

    dataset_sizes = {}
    if "dataset" in df.columns:
        dataset_sizes = df["dataset"].value_counts().to_dict()

    return {
        "missing_counts": missing_counts,
        "dataset_sizes": dataset_sizes,
        "errors": errors
    }


def split_by_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by dataset column.

    Args:
        df: DataFrame with a 'dataset' column.

    Returns:
        Tuple of (mmlu_df, gsm8k_df).
    """
    mmlu_df = df[df["dataset"] == "mmlu"].copy()
    gsm8k_df = df[df["dataset"] == "gsm8k"].copy()
    return mmlu_df, gsm8k_df


def clean_mmlu_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Mark MMLU questions with unknown answers as incorrect.

    For MMLU questions where model_answer="?", sets correct=False.
    Does not remove rows, only modifies the correct column.

    Args:
        df: DataFrame with 'dataset', 'model_answer', and 'correct' columns.

    Returns:
        Modified DataFrame with corrected values for "?" answers.
    """
    df = df.copy()

    # Find MMLU rows with "?" model_answer
    mask = (df["dataset"] == "mmlu") & (df["model_answer"] == "?")
    df.loc[mask, "correct"] = False

    return df