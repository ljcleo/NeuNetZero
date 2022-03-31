from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @ staticmethod
    @ abstractmethod
    def loss(pred: np.ndarray, label: np.ndarray) -> float:
        pass

    @ staticmethod
    @ abstractmethod
    def gradient(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
        pass


class SoftmaxNLL(Loss):
    @staticmethod
    def loss(pred: np.ndarray, label: np.ndarray) -> float:
        max_pred: np.ndarray = pred.max(axis=1, keepdims=True)
        return np.mean(np.log(np.exp(pred - max_pred).sum(axis=1)) + max_pred.ravel()
                       - pred[np.arange(pred.shape[0]), label])

    @staticmethod
    def gradient(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
        exp_pred: np.ndarray = np.exp(pred - pred.max(axis=1, keepdims=True))
        grad: np.ndarray = exp_pred / exp_pred.sum(axis=1, keepdims=True)
        grad[np.arange(pred.shape[0]), label] -= 1
        return grad
