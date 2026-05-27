"""Tests for extraction_schema — schema validation for extraction output."""

import numpy as np
import pytest
from src.extraction_schema import (
    validate_extraction_schema,
    HIDDEN_DIM,
    REQUIRED_LAYER_KEYS,
    REQUIRED_SCALAR_KEYS,
    ALL_REQUIRED_KEYS,
)


class TestSchemaConstants:
    def test_hidden_dim_is_2560(self):
        assert HIDDEN_DIM == 2560

    def test_required_layers_match_spec(self):
        # Pre-registered layers as per hybrid architecture
        assert REQUIRED_LAYER_KEYS == [
            "layers_15", "layers_17", "layers_19",
            "layers_20", "layers_25",
        ]

    def test_all_required_keys_present(self):
        assert set(ALL_REQUIRED_KEYS) == {
            "layers_15", "layers_17", "layers_19", "layers_20", "layers_25",
            "msp", "entropy", "generated_text", "correctness", "split",
        }


class TestValidateExtractionSchema:
    def make_valid_dict(self) -> dict:
        """Helper to build a valid extraction dict."""
        return {
            "layers_15": np.zeros((1, 2560)),
            "layers_17": np.zeros((1, 2560)),
            "layers_19": np.zeros((1, 2560)),
            "layers_20": np.zeros((1, 2560)),
            "layers_25": np.zeros((1, 2560)),
            "msp": 0.95,
            "entropy": 0.5,
            "generated_text": "The answer is 42.",
            "correctness": True,
            "split": "train",
        }

    def test_valid_dict_passes(self):
        data = self.make_valid_dict()
        assert validate_extraction_schema(data) is True

    def test_missing_layer_key_fails(self):
        data = self.make_valid_dict()
        del data["layers_15"]
        assert validate_extraction_schema(data) is False

    def test_missing_scalar_key_fails(self):
        data = self.make_valid_dict()
        del data["msp"]
        assert validate_extraction_schema(data) is False

    def test_wrong_layer_shape_fails(self):
        data = self.make_valid_dict()
        data["layers_15"] = np.zeros((1, 128))
        assert validate_extraction_schema(data) is False

    def test_layer_not_numpy_fails(self):
        data = self.make_valid_dict()
        data["layers_17"] = [[0.0] * 2560]
        assert validate_extraction_schema(data) is False

    def test_correctness_must_be_bool(self):
        data = self.make_valid_dict()
        data["correctness"] = "True"
        assert validate_extraction_schema(data) is False

    def test_split_must_be_train_or_val(self):
        data = self.make_valid_dict()
        data["split"] = "test"
        assert validate_extraction_schema(data) is False

    def test_val_split_is_valid(self):
        data = self.make_valid_dict()
        data["split"] = "val"
        assert validate_extraction_schema(data) is True

    def test_generated_text_must_be_string(self):
        data = self.make_valid_dict()
        data["generated_text"] = 12345
        assert validate_extraction_schema(data) is False
