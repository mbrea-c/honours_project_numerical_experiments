from locale import Error
from typing import List, Tuple, Callable
import numpy as np
from scipy.special import xlogy
from common.funcs import softmax
from common import Model
import logging


class Activation:
    def __call__(self, input):
        raise NotImplementedError()

    def grad(self, input):
        raise NotImplementedError()


class Cost:
    def __call__(self, input, labels):
        raise NotImplementedError()

    def grad(self, input, labels):
        raise NotImplementedError()


class Regularizer:
    def __call__(self, input, labels):
        raise NotImplementedError()

    def grad(self, input, labels):
        raise NotImplementedError()


class CrossEntropy(Cost):
    def __call__(self, predictions, labels):
        cost = -np.sum(xlogy(labels, predictions)) / predictions.shape[0]  # Scalar
        return cost

    def grad(self, predictions, labels):
        return labels / predictions / predictions.shape[0]


class L2(Cost):
    def __call__(self, predictions, labels):
        cost = (
            np.sum(np.square((predictions - labels))) / predictions.shape[0]
        )  # Scalar
        return cost

    def grad(self, predictions, labels):
        return 2 * (predictions - labels) / predictions.shape[0]


class Softmax(Activation):
    def __call__(self, input):
        return softmax(input)


class Sigmoid(Activation):
    def __call__(self, input):
        return 1 / (1 + np.exp(-input))

    def grad(self, input):
        f = self.__call__(input)
        return f * (1 - f)


Dimension = int
LayerParams = Tuple[Dimension, Activation]


class Layer:
    def __init__(
        self,
        theta: np.ndarray,
        bias: np.ndarray,
        activation_func: Activation = Softmax(),
    ):
        self.theta = theta
        self.bias = bias
        self.activation_func = activation_func

        self.last_input: np.ndarray | None = None
        self.can_train: bool = False

    def forward(self, input: np.ndarray) -> np.ndarray:
        a = input @ self.theta.T + np.ones((input.shape[0], 1)) @ self.bias.T
        z = self.activation_func(a)
        self.last_a = a
        self.last_input = input
        self.can_train = True
        return z

    def backward(
        self,
        a_grad_given: np.ndarray | None,
        z_grad_given: np.ndarray | None,
        labels: np.ndarray,
        learning_rate: float,
        regularization_param: float,
    ) -> np.ndarray:
        if self.can_train == False:
            logging.error(
                "Cannot perform backward step; try calling <Model>.forward(...) first"
            )
            raise Exception(
                "Cannot perform backward step; try calling <Model>.forward(...) first"
            )
        else:
            if a_grad_given is not None:
                a_grad: np.ndarray = a_grad_given
            elif z_grad_given is not None:
                a_grad: np.ndarray = (
                    self.activation_func.grad(self.last_a) * z_grad_given
                )
            else:
                raise Exception("Neither a_grad nor z_grad are given")
            theta_grad = a_grad.T @ self.last_input
            bias_grad = a_grad.T @ np.ones(shape=(self.last_input.shape[0], 1))
            prev_layer_z_grad = a_grad @ self.theta

            assert theta_grad.shape == self.theta.shape
            assert bias_grad.shape == self.bias.shape

            self.theta -= learning_rate * (
                theta_grad + regularization_param * self.theta
            )
            self.bias -= learning_rate * (bias_grad + regularization_param * self.bias)
            self.can_train = False

            return prev_layer_z_grad


class NeuralNetwork(Model):
    def __init__(
        self,
        input_dim,
        layer_params: List[LayerParams],
        cost_fn: Cost,
        regularization_param=0.01,
    ):
        self.can_train: bool = False
        self.layers: List[Layer] = []
        self.cost_fn = cost_fn
        self.last_f = None
        self.regularization_param = regularization_param

        in_dim = input_dim
        for out_dim, activation_func in layer_params:
            theta, bias = self.__initializer__((out_dim, in_dim))
            layer = Layer(theta, bias, activation_func)
            self.layers.append(layer)
            in_dim = out_dim

    def forward(self, input: np.ndarray) -> np.ndarray:
        z = input
        for layer in self.layers:
            z = layer.forward(z)
        self.can_train = True
        self.last_f = z
        return z

    def backward(self, labels: np.ndarray, learning_rate: float) -> None:
        if self.can_train == False:
            logging.error(
                "Cannot perform backward step; try calling <Model>.forward(...) first"
            )
            raise Exception(
                "Cannot perform backward step; try calling <Model>.forward(...) first"
            )
        else:
            if (
                type(self.cost_fn) is CrossEntropy
                and type(self.layers[-1].activation_func) is Softmax
            ):
                a_grad = self.last_f - labels
                z_grad = self.layers[-1].backward(
                    a_grad_given=a_grad,
                    z_grad_given=None,
                    labels=labels,
                    learning_rate=learning_rate,
                    regularization_param=self.regularization_param,
                )
                for layer in reversed(self.layers[:-1]):
                    z_grad = layer.backward(
                        a_grad_given=None,
                        z_grad_given=z_grad,
                        labels=labels,
                        learning_rate=learning_rate,
                        regularization_param=self.regularization_param,
                    )
            elif type(self.cost_fn) is L2:
                z_grad = self.cost_fn.grad(self.last_f, labels)
                for layer in reversed(self.layers):
                    z_grad = layer.backward(
                        a_grad_given=None,
                        z_grad_given=z_grad,
                        labels=labels,
                        learning_rate=learning_rate,
                    )
            else:
                raise Exception("Output layer/cost combination not allowed")
            self.can_train = False

    def __initializer__(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        N_out, N_in = shape
        theta = np.random.normal(loc=0, scale=1 / (np.sqrt(N_in)), size=shape)
        bias = np.zeros(shape=(shape[0], 1))
        return theta, bias

    def cost(self, input: np.ndarray, labels: np.ndarray):
        """Negative log-likelihood, or cross-entropy"""
        f = self.forward(input)
        cost = self.cost_fn(f, labels)  # Scalar
        return cost

    def set_weights(self, weights) -> None:
        for layer, weights in zip(self.layers, weights):
            layer.theta = np.copy(weights[0])
            layer.bias = np.copy(weights[1])

    def get_weights(self):
        return [(np.copy(layer.theta), np.copy(layer.bias)) for layer in self.layers]
