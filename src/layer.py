from abc import ABC, abstractmethod

import numpy as np

from initializer import Constant, Gaussian, Initializer


class Module(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, e: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params_and_grads(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        pass

    @abstractmethod
    def toggle_train(self, train: bool) -> None:
        pass


class Flatten(Module):
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.original_shape: tuple[int, ...] = x.shape
        return x.reshape(x.shape[0], -1) if len(self.original_shape) > 1 else x

    def backward(self, e: np.ndarray) -> np.ndarray:
        return e.reshape(self.original_shape)

    def get_params_and_grads(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([]), np.array([])

    def set_params(self, params: np.ndarray) -> None:
        pass

    def toggle_train(self, train: bool) -> None:
        pass


class Linear(Module):
    def __init__(self, n_input: int, n_output: int, use_bias: bool = True,
                 weight_init: Initializer = Gaussian(),
                 bias_init: Initializer = Constant()) -> None:
        self.use_bias: bool = use_bias
        self.weight: np.ndarray = weight_init.initialize((n_input, n_output))
        self.weight_grad: np.ndarray = np.zeros_like(self.weight)

        if self.use_bias:
            self.bias: np.ndarray = bias_init.initialize((1, n_output))
            self.bias_grad: np.ndarray = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.data_in: np.ndarray = x
        data_out: np.ndarray = x @ self.weight

        if self.use_bias:
            data_out += self.bias
        return data_out

    def backward(self, e: np.ndarray) -> np.ndarray:
        self.weight_grad += self.data_in.T @ e
        if self.use_bias:
            self.bias_grad += e.sum(axis=0, keepdims=True)
        return e @ self.weight.T

    def get_params_and_grads(self) -> tuple[np.ndarray, np.ndarray]:
        if self.use_bias:
            params: np.ndarray = np.concatenate((self.weight.ravel(), self.bias.ravel()))
            grads: np.ndarray = np.concatenate((self.weight_grad.ravel(), self.bias_grad.ravel()))
            return params, grads
        else:
            return self.weight.ravel(), self.weight_grad.ravel()

    def set_params(self, params: np.ndarray) -> None:
        if self.use_bias:
            weight_size: int = self.weight.shape[0] * self.weight.shape[1]
            self.weight = params[:weight_size].reshape(self.weight.shape)
            self.bias = params[weight_size:].reshape(self.bias.shape)
            self.weight_grad = np.zeros_like(self.weight)
            self.bias_grad = np.zeros_like(self.bias)
        else:
            self.weight = params.reshape(self.weight.shape)
            self.weight_grad = np.zeros_like(self.weight)

    def toggle_train(self, train: bool) -> None:
        pass


class ReLU(Module):
    def __init__(self, alpha: float = 0) -> None:
        self.alpha: float = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.grad: np.ndarray = x >= 0
        self.grad[~self.grad] = self.alpha
        return x * self.grad

    def backward(self, e: np.ndarray) -> np.ndarray:
        return e * self.grad

    def get_params_and_grads(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([]), np.array([])

    def get_state(self) -> np.ndarray:
        return np.array([])

    def set_params(self, params: np.ndarray) -> None:
        pass

    def toggle_train(self, train: bool) -> None:
        pass


class Dropout(Module):
    def __init__(self, rate: float) -> None:
        self.rate: float = rate
        self.train: bool = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        shape: tuple[int, ...] = (1, ) + x.shape[1:]
        self.mask: np.ndarray = (np.random.rand(*shape) > self.rate) / (1 - self.rate) \
            if self.train else np.ones(shape, dtype=np.int32)
        return x * self.mask

    def backward(self, e: np.ndarray) -> np.ndarray:
        return e * self.mask

    def get_params_and_grads(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([]), np.array([])

    def get_state(self) -> np.ndarray:
        return np.array([])

    def set_params(self, params: np.ndarray) -> None:
        pass

    def set_state(self, state: np.ndarray) -> None:
        pass

    def toggle_train(self, train: bool) -> None:
        self.train = train
