from pathlib import Path
from pickle import dump, load

import numpy as np

from initializer import Gaussian
from layer import Flatten, Linear, Module, ReLU


class Model:
    def __init__(self, modules: list[Module]) -> None:
        self.modules: list[Module] = modules
        self.param_size: list[int] = [module.get_params_and_grads()[0].shape[0]
                                      for module in self.modules]

        self.batch_size: int = 0

    @property
    def fine_tune_mask(self) -> np.ndarray:
        mask: np.ndarray = np.zeros(sum(self.param_size))
        mask[-self.param_size[-1]:] = 1
        return mask

    def forward(self, x: np.ndarray) -> np.ndarray:
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, e: np.ndarray) -> np.ndarray:
        self.batch_size += e.shape[0]
        for module in reversed(self.modules):
            e = module.backward(e)
        return e

    def get_params_and_grads(self) -> tuple[np.ndarray, np.ndarray]:
        params: list[np.ndarray] = []
        grads: list[np.ndarray] = []

        for module in self.modules:
            param, grad = module.get_params_and_grads()
            params.append(param)
            grads.append(grad)

        return np.concatenate(params), np.concatenate(grads) / (self.batch_size + 1e-8)

    def set_params(self, params: np.ndarray) -> None:
        offset: int = 0
        self.batch_size = 0

        for index, module in enumerate(self.modules):
            length: int = self.param_size[index]
            module.set_params(params[offset: offset + length])
            offset += length

    def check_nan(self) -> bool:
        params: np.ndarray = self.get_params_and_grads()[0]
        return np.any(np.logical_or(np.isinf(params), np.isnan(params)))

    def load_params(self, path: Path) -> None:
        with path.open('rb') as f:
            self.set_params(load(f))

    def dump_params(self, path: Path) -> None:
        with path.open('wb') as f:
            dump(self.get_params_and_grads()[0], f)


class ImageClsMLP(Model):
    def __init__(self, image_size: tuple[int, int], n_class: int, hidden: list[int]) -> None:
        layers: list[Module] = [Flatten(), Linear(image_size[0] * image_size[1],
                                                  hidden[0], weight_init=Gaussian(0, 0.1))]
        for i in range(len(hidden) - 1):
            layers.extend([ReLU(), Linear(hidden[i], hidden[i + 1], weight_init=Gaussian(0, 0.1))])

        layers.extend([ReLU(), Linear(hidden[-1], n_class, weight_init=Gaussian(0, 0.1))])
        return super(ImageClsMLP, self).__init__(layers)
