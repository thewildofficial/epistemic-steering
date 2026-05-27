"""Mean pooling over the token dimension.

Reduces a sequence of hidden states (seq_len, hidden_dim) to a single
vector (hidden_dim,) by averaging across tokens.

Functions:
- mean_pool: Average hidden states across sequence dimension
"""

import numpy as np


def mean_pool(hidden_states: np.ndarray) -> np.ndarray:
    """Mean pool hidden states over the sequence dimension.

    Args:
        hidden_states: numpy array of shape (seq_len, hidden_dim).

    Returns:
        numpy array of shape (hidden_dim,) with mean values.
        Returns zero vector if seq_len is 0.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            f"Expected 2D array (seq_len, hidden_dim), got shape {hidden_states.shape}"
        )

    if hidden_states.shape[0] == 0:
        return np.zeros(hidden_states.shape[1], dtype=hidden_states.dtype)

    return np.mean(hidden_states, axis=0)
