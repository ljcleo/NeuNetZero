from pathlib import Path
from typing import Optional

import numpy as np

from ..data.loader import DataLoader
from ..nn.model import Model


def evaluate(model: Model, loader: DataLoader) -> float:
    model.toggle_train(False)
    all_pred: list[np.ndarray] = []
    all_label: list[np.ndarray] = []

    for input_batch, output_batch in loader:
        all_pred.append(model.forward(input_batch))
        all_label.append(output_batch)

    pred: np.ndarray = np.argmax(np.concatenate(all_pred), axis=1)
    label: np.ndarray = np.concatenate(all_label)
    return (pred == label).mean()


def get_path(root_path: Path, sub_dir: str, name: Optional[str] = None) -> Path:
    result: Path = root_path / sub_dir
    if name is not None:
        result = result / name.split('-')[0]

    result.mkdir(parents=True, exist_ok=True)
    return result
