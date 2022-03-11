import numpy as np


def softmax(mat_input: np.ndarray) -> np.ndarray:
    """
    Takes input (N x K) vector X (where each row is one example), and
    returns the output (N x K) matrix
    """
    shifted_input = mat_input - np.max(mat_input, axis=1, keepdims=True)
    s = np.exp(shifted_input)
    sum_s = np.sum(s, axis=1, keepdims=True)
    f = np.divide(s, sum_s, dtype=np.float64)
    return f
