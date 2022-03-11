import numpy as np
from typing import Tuple


class Model:
    def forward(self, input: np.ndarray):
        raise NotImplementedError()

    def backward(self, labels: np.ndarray, learning_rate: float):
        raise NotImplementedError()

    def cost(self, input: np.ndarray, labels: np.ndarray):
        raise NotImplementedError()

    def get_weights(self) -> np.ndarray:
        raise NotImplementedError()

    def set_weights(self, weights: np.ndarray) -> None:
        raise NotImplementedError()

    def pred_accuracy(self, input: np.ndarray, labels: np.ndarray):
        f = self.forward(input)
        f_pred = np.argmax(f, axis=1)
        y_label = np.argmax(labels, axis=1)

        n_correct = np.sum((f_pred == y_label).astype(np.int64))
        return n_correct / input.shape[0]
