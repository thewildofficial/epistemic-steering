"""Tests for held-out evaluation scripts (static functions only).

These tests verify helper functions without requiring Modal or GPU.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate_heldout import (
    check_correctness,
    extract_answer_gsm8k,
    extract_answer_mmlu,
    format_gsm8k_prompt,
    format_mmlu_prompt,
)
from evaluate_baseline_direct import (
    check_correctness as check_direct,
    extract_answer_gsm8k as extract_gsm_direct,
    extract_answer_mmlu as extract_mmlu_direct,
)


class TestFormatMMLUPrompt:
    def test_basic_formatting(self):
        example = {
            "question": "What is 2+2?",
            "choices": ["1", "2", "3", "4"],
            "answer": "D",
        }
        prompt = format_mmlu_prompt(example)
        assert "What is 2+2?" in prompt
        assert "A) 1" in prompt
        assert "B) 2" in prompt
        assert "C) 3" in prompt
        assert "D) 4" in prompt
        assert "Answer:" in prompt

    def test_letter_extraction(self):
        assert extract_answer_mmlu("A") == "A"
        assert extract_answer_mmlu("  B  ") == "B"
        assert extract_answer_mmlu("The answer is C.") == "C"
        assert extract_answer_mmlu("I think D is correct") == "D"
        assert extract_answer_mmlu("") is None
        assert extract_answer_mmlu("No answer here") is None


class TestFormatGSM8KPrompt:
    def test_basic_formatting(self):
        example = {"question": "What is 2+2?"}
        prompt = format_gsm8k_prompt(example)
        assert "What is 2+2?" in prompt
        assert "Solve this math problem" in prompt
        assert "Answer:" in prompt

    def test_number_extraction(self):
        assert extract_answer_gsm8k("42") == "42"
        assert extract_answer_gsm8k("The answer is 123.") == "123"
        assert extract_answer_gsm8k("Step 1: 5, Step 2: 10, final: 15") == "15"
        assert extract_answer_gsm8k("-3.5") == "-3.5"
        assert extract_answer_gsm8k("3.14159") == "3.14159"
        assert extract_answer_gsm8k("") is None
        assert extract_answer_gsm8k("No numbers") is None


class TestCheckCorrectness:
    def test_mmlu_correct(self):
        assert check_correctness("A", "A", "mmlu") is True
        assert check_correctness("a", "A", "mmlu") is True

    def test_mmlu_incorrect(self):
        assert check_correctness("B", "A", "mmlu") is False

    def test_gsm8k_numeric(self):
        assert check_correctness("42", "42", "gsm8k") is True
        assert check_correctness("42.0", "42", "gsm8k") is True
        assert check_correctness("42", "42.0", "gsm8k") is True

    def test_gsm8k_string_fallback(self):
        assert check_correctness("forty-two", "forty-two", "gsm8k") is True
        assert check_correctness("forty-two", "42", "gsm8k") is False

    def test_none_prediction(self):
        assert check_correctness(None, "A", "mmlu") is False

    def test_direct_imports(self):
        # Verify baseline scripts have same helpers
        assert extract_mmlu_direct("C") == "C"
        assert extract_gsm_direct("99") == "99"
        assert check_direct("D", "D", "mmlu") is True


class TestDataLeakagePrevention:
    def test_train_prompts_loaded(self, tmp_path):
        # Create a mock training file
        train_path = tmp_path / "train.jsonl"
        with open(train_path, "w") as f:
            f.write('{"question_id": "q1", "prompt": "prompt1"}\n')
            f.write('{"question_id": "q2", "prompt": "prompt2"}\n')

        from evaluate_heldout import load_training_data
        train_ids, train_prompts = load_training_data(str(train_path))
        assert train_ids == {"q1", "q2"}
        assert train_prompts == {"prompt1", "prompt2"}

    @pytest.mark.skip(reason="Requires datasets library and network access")
    def test_prompt_deduplication(self, tmp_path):
        from evaluate_heldout import load_training_data, load_heldout_questions

        train_path = tmp_path / "train.jsonl"
        with open(train_path, "w") as f:
            f.write('{"question_id": "mmlu_x_0", "prompt": "train_prompt"}\n')

        train_ids, train_prompts = load_training_data(str(train_path))
        # Mock heldout with same prompt should be filtered
        heldout = load_heldout_questions(train_ids, train_prompts, 10)
        # Since we can't load real datasets in test, heldout should be empty
        # or only contain questions not matching train_prompts
        for q in heldout:
            assert q["prompt"] not in train_prompts


class TestScriptStructure:
    def test_evaluate_heldout_has_modal_function(self):
        import evaluate_heldout
        assert hasattr(evaluate_heldout, "evaluate_heldout")
        assert hasattr(evaluate_heldout, "app")

    def test_baseline_direct_has_modal_function(self):
        import evaluate_baseline_direct
        assert hasattr(evaluate_baseline_direct, "evaluate_baseline_direct")
        assert hasattr(evaluate_baseline_direct, "app")

    def test_baseline_cot_has_modal_function(self):
        import evaluate_baseline_cot
        assert hasattr(evaluate_baseline_cot, "evaluate_baseline_cot")
        assert hasattr(evaluate_baseline_cot, "app")

    def test_baseline_random_has_modal_function(self):
        import evaluate_baseline_random
        assert hasattr(evaluate_baseline_random, "evaluate_baseline_random")
        assert hasattr(evaluate_baseline_random, "app")
