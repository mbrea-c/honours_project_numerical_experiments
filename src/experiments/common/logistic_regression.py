import numpy as np
from typing import Tuple
from common.funcs import softmax
from common import Model
import logging


class LogisticRegression(Model):
    def __init__(self, weights_init):
        self.weights: np.ndarray = weights_init
        self.last_input = None
        self.last_f = None
        self.can_train = False

    def forward(self, input: np.ndarray) -> np.ndarray:
        s = input @ self.weights.T  # N x K
        f = softmax(s)  # N x K

        self.last_input = input
        self.last_f = f
        self.can_train = True
        return self.last_f

    def backward(self, labels: np.ndarray, learning_rate: float) -> None:
        if self.can_train == False:
            logging.error(
                "Cannot perform backward step; try calling <Model>.forward(...) first"
            )
        else:
            err = labels - self.last_f
            grads = -err.T @ self.last_input
            self.weights -= learning_rate * grads
            self.can_train = False

    def cost(self, input: np.ndarray, labels: np.ndarray):
        """Negative log-likelihood"""
        m = input @ self.weights.T  # N x K
        s = np.exp(m - np.max(m, axis=1, keepdims=True))  # N x K
        sum_s = np.sum(s, axis=1, keepdims=True)  # N x 1

        cost = np.sum(labels * (np.log(sum_s) - m))  # Scalar

        return cost

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = np.copy(weights)

    def get_weights(self) -> np.ndarray:
        return np.copy(self.weights)
