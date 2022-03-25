from abc import ABC, abstractmethod

import numpy as np


class Initializer(ABC):
    @abstractmethod
    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        pass


class Constant(Initializer):
    def __init__(self, constant: float = 0) -> None:
        self.constant: float = constant

    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.ones(shape) * self.constant


class Gaussian(Initializer):
    def __init__(self, mean: float = 0, std: float = 1) -> None:
        self.mean = mean
        self.std = std

    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.random.normal(self.mean, self.std, shape)
