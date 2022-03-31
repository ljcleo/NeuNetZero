from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .model import Model


class Optimizer(ABC):
    @ abstractmethod
    def update(self) -> None:
        pass

    @ abstractmethod
    def set_multiplier(self, multiplier: float) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, model: Model, lr: float, l2_lambda: float = 0) -> None:
        self.model: Model = model
        self.lr: float = lr
        self.l2_lambda: float = l2_lambda
        self.multiplier: float = 1

    def update(self) -> None:
        params: np.ndarray
        grads: np.ndarray
        params, grads = self.model.get_params_and_grads()
        grads += 2 * self.l2_lambda * params
        self.model.set_params(params - self.lr * grads * self.multiplier)

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = multiplier


class MomentumSGD(Optimizer):
    def __init__(self, model: Model, lr: float, beta: float = 0.9, l2_lambda: float = 0) -> None:
        self.model: Model = model
        self.lr: float = lr
        self.beta: float = beta
        self.l2_lambda: float = l2_lambda
        self.multiplier: float = 1

        self._prev: Optional[np.ndarray] = None

    def update(self) -> None:
        params: np.ndarray
        grads: np.ndarray
        params, grads = self.model.get_params_and_grads()
        grads += 2 * self.l2_lambda * params

        if self._prev is None:
            self._prev = params

        self.model.set_params(
            params - (self.lr * grads - self.beta * (params - self._prev)) * self.multiplier
        )
        self._prev = params

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = multiplier
