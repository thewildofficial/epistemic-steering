"""Extraction schema for hidden-state extraction pipeline.

Defines the required keys and types for the extraction output dict.
Used to validate that extraction results have the expected structure
before downstream processing.

Functions:
- validate_extraction_schema: Validate a data dict against the schema
"""

import numpy as np

# Hidden dimension for Qwen3.5-4B
HIDDEN_DIM: int = 2560

# Required layer keys — pre-registered layers
REQUIRED_LAYER_KEYS: list[str] = [
    "layers_15",
    "layers_17",
    "layers_19",
    "layers_20",
    "layers_25",
]

# Required scalar/string/boolean keys
REQUIRED_SCALAR_KEYS: list[str] = [
    "msp",
    "entropy",
    "generated_text",
    "correctness",
    "split",
]

# All required keys combined
ALL_REQUIRED_KEYS: list[str] = REQUIRED_LAYER_KEYS + REQUIRED_SCALAR_KEYS


def validate_extraction_schema(data: dict) -> bool:
    """Validate that a data dict conforms to the extraction schema.

    Checks:
    - All required keys are present.
    - Layer keys are numpy arrays with shape (1, HIDDEN_DIM).
    - Scalar keys have correct types:
        msp: float, entropy: float, generated_text: str,
        correctness: bool, split: str (one of 'train', 'val').

    Args:
        data: Dict with extraction results.

    Returns:
        True if valid, False otherwise.
    """
    # Check all keys present
    for key in ALL_REQUIRED_KEYS:
        if key not in data:
            return False

    # Check layer arrays
    for key in REQUIRED_LAYER_KEYS:
        val = data[key]
        if not isinstance(val, np.ndarray):
            return False
        if val.shape != (1, HIDDEN_DIM):
            return False

    # Check scalar types
    if not isinstance(data["msp"], (int, float, np.floating)):
        return False
    if not isinstance(data["entropy"], (int, float, np.floating)):
        return False
    if not isinstance(data["generated_text"], str):
        return False
    if not isinstance(data["correctness"], (bool, np.bool_)):
        return False
    if not isinstance(data["split"], str):
        return False
    if data["split"] not in ("train", "val"):
        return False

    return True
