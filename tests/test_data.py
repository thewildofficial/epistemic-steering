"""Tests for data loading and validation functions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestLoadProbeResults:
    """Tests for load_probe_results function."""

    def test_load_probe_results_basic(self, tmp_path):
        jsonl_file = tmp_path / "probe_results.jsonl"
        data = [
            {
                "question_id": 1,
                "dataset": "mmlu",
                "prompt": "What is the capital of France?",
                "generated_text": "Paris is the capital.",
                "model_answer": "Paris",
                "correct": True,
                "top_token_prob": 0.95
            },
            {
                "question_id": 2,
                "dataset": "gsm8k",
                "prompt": "If John has 5 apples...",
                "generated_text": "10 apples total.",
                "model_answer": "10",
                "correct": False,
                "top_token_prob": 0.72
            }
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in data))

        from src.data import load_probe_results
        df = load_probe_results(str(jsonl_file))

        assert len(df) == 2
        assert df["question_id"].dtype == int
        assert df["correct"].dtype == bool
        assert df["question_id"].tolist() == [1, 2]
        assert df["dataset"].tolist() == ["mmlu", "gsm8k"]

    def test_load_probe_results_empty_file(self, tmp_path):
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        from src.data import load_probe_results
        df = load_probe_results(str(jsonl_file))

        assert len(df) == 0


class TestLoadAllposResults:
    """Tests for load_allpos_results function."""

    def test_load_allpos_results_basic(self, tmp_path):
        jsonl_file = tmp_path / "allpos_results.jsonl"
        data = [
            {
                "question_id": 1,
                "dataset": "mmlu",
                "prompt": "What is 2+2?",
                "generated_text": "4",
                "model_answer": "4",
                "correct": True,
                "top_token_prob": 0.98,
                "token_positions": [0, 1, 2, 3]
            },
            {
                "question_id": 2,
                "dataset": "gsm8k",
                "prompt": "Solve for x...",
                "generated_text": "x=5",
                "model_answer": "5",
                "correct": True,
                "top_token_prob": 0.88,
                "token_positions": [0, 1, 2]
            }
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in data))

        from src.data import load_allpos_results
        df = load_allpos_results(str(jsonl_file))

        assert len(df) == 2
        assert "token_positions" in df.columns
        assert df.loc[0, "token_positions"] == [0, 1, 2, 3]
        assert df.loc[1, "token_positions"] == [0, 1, 2]


class TestLoadActivations:
    """Tests for load_activations function."""

    def test_load_activations_basic(self, tmp_path):
        activations_dir = tmp_path / "activations"
        activations_dir.mkdir()
        arr = np.random.randn(128, 2560).astype(np.float32)
        np.save(activations_dir / "q42_layer7.npy", arr)

        from src.data import load_activations
        result = load_activations(str(activations_dir), 42, 7)

        assert isinstance(result, np.ndarray)
        assert result.shape == (128, 2560)
        assert result.dtype == np.float32

    def test_load_activations_file_not_found(self, tmp_path):
        from src.data import load_activations
        with pytest.raises(FileNotFoundError):
            load_activations(str(tmp_path), 999, 0)


class TestValidateData:
    """Tests for validate_data function."""

    def test_validate_data_valid(self):
        from src.data import validate_data
        df = pd.DataFrame({
            "question_id": [1, 2, 3],
            "dataset": ["mmlu", "gsm8k", "mmlu"],
            "prompt": ["Q1", "Q2", "Q3"],
            "generated_text": ["A1", "A2", "A3"],
            "model_answer": ["A", "B", "C"],
            "correct": [True, False, True],
            "top_token_prob": [0.9, 0.8, 0.85]
        })

        result = validate_data(df)

        assert result["missing_counts"]["question_id"] == 0
        assert result["dataset_sizes"]["mmlu"] == 2
        assert result["dataset_sizes"]["gsm8k"] == 1
        assert result["errors"] == []

    def test_validate_data_missing_values(self):
        from src.data import validate_data
        df = pd.DataFrame({
            "question_id": [1, None, 3],
            "dataset": ["mmlu", "gsm8k", "mmlu"],
            "prompt": ["Q1", "Q2", None],
            "generated_text": ["A1", "A2", "A3"],
            "model_answer": ["A", "B", "C"],
            "correct": [True, False, True],
            "top_token_prob": [0.9, 0.8, 0.85]
        })

        result = validate_data(df)

        assert result["missing_counts"]["question_id"] == 1
        assert result["missing_counts"]["prompt"] == 1
        assert len(result["errors"]) > 0

    def test_validate_data_invalid_dataset(self):
        from src.data import validate_data
        df = pd.DataFrame({
            "question_id": [1, 2],
            "dataset": ["mmlu", "invalid_dataset"],
            "prompt": ["Q1", "Q2"],
            "generated_text": ["A1", "A2"],
            "model_answer": ["A", "B"],
            "correct": [True, False],
            "top_token_prob": [0.9, 0.8]
        })

        result = validate_data(df)

        assert len(result["errors"]) > 0
        assert any("invalid_dataset" in err for err in result["errors"])

    def test_validate_data_correct_not_bool(self):
        from src.data import validate_data
        df = pd.DataFrame({
            "question_id": [1, 2],
            "dataset": ["mmlu", "gsm8k"],
            "prompt": ["Q1", "Q2"],
            "generated_text": ["A1", "A2"],
            "model_answer": ["A", "B"],
            "correct": ["True", "False"],
            "top_token_prob": [0.9, 0.8]
        })

        result = validate_data(df)

        assert len(result["errors"]) > 0


class TestSplitByDataset:
    """Tests for split_by_dataset function."""

    def test_split_by_dataset_basic(self):
        from src.data import split_by_dataset
        df = pd.DataFrame({
            "question_id": [1, 2, 3, 4],
            "dataset": ["mmlu", "gsm8k", "mmlu", "gsm8k"],
            "prompt": ["Q1", "Q2", "Q3", "Q4"],
            "generated_text": ["A1", "A2", "A3", "A4"],
            "model_answer": ["A", "B", "C", "D"],
            "correct": [True, False, True, False],
            "top_token_prob": [0.9, 0.8, 0.85, 0.75]
        })

        mmlu_df, gsm8k_df = split_by_dataset(df)

        assert len(mmlu_df) == 2
        assert len(gsm8k_df) == 2
        assert mmlu_df["dataset"].unique()[0] == "mmlu"
        assert gsm8k_df["dataset"].unique()[0] == "gsm8k"

    def test_split_by_dataset_mmlu_only(self):
        from src.data import split_by_dataset
        df = pd.DataFrame({
            "question_id": [1, 2],
            "dataset": ["mmlu", "mmlu"],
            "prompt": ["Q1", "Q2"],
            "generated_text": ["A1", "A2"],
            "model_answer": ["A", "B"],
            "correct": [True, False],
            "top_token_prob": [0.9, 0.8]
        })

        mmlu_df, gsm8k_df = split_by_dataset(df)

        assert len(mmlu_df) == 2
        assert len(gsm8k_df) == 0


class TestCleanMmluAnswers:
    """Tests for clean_mmlu_answers function."""

    def test_clean_mmlu_answers_question_mark(self):
        from src.data import clean_mmlu_answers
        df = pd.DataFrame({
            "question_id": [1, 2, 3],
            "dataset": ["mmlu", "mmlu", "gsm8k"],
            "model_answer": ["A", "?", "42"],
            "correct": [True, True, True]
        })

        result = clean_mmlu_answers(df)

        assert result.loc[1, "correct"] == False
        assert result.loc[0, "correct"] == True
        assert result.loc[2, "correct"] == True
        assert len(result) == 3

    def test_clean_mmlu_answers_no_change_needed(self):
        from src.data import clean_mmlu_answers
        df = pd.DataFrame({
            "question_id": [1, 2],
            "dataset": ["mmlu", "gsm8k"],
            "model_answer": ["A", "B"],
            "correct": [True, False]
        })

        result = clean_mmlu_answers(df)

        assert result.loc[0, "correct"] == True
        assert result.loc[1, "correct"] == False

    def test_clean_mmlu_answers_preserves_rows(self):
        from src.data import clean_mmlu_answers
        df = pd.DataFrame({
            "question_id": [1, 2, 3],
            "dataset": ["mmlu", "mmlu", "mmlu"],
            "model_answer": ["A", "?", "?"],
            "correct": [True, True, True]
        })

        result = clean_mmlu_answers(df)

        assert len(result) == 3