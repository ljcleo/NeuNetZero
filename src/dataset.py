from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Literal, Union

import numpy as np


class Dataset(ABC):
    @ abstractmethod
    def __len__(self) -> int:
        pass

    @ abstractmethod
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        pass


class MNISTDataset(Dataset):
    def __init__(self, data_root_folder: Path, mode: Literal['training', 'test']) -> None:
        if mode not in ['training', 'test']:
            raise ValueError('invalid mode')

        folder: Path = data_root_folder / mode
        self.n_data: int = 60000 if mode == 'training' else 10000
        self.image_size: tuple[int, int] = (28, 28)

        with (folder / 'images.bin').open('rb') as f:
            if f.read(4) != b'\x00\x00\x08\x03':
                raise RuntimeError('corrupted image data')
            if self._byte_to_int(f.read(4)) != self.n_data:
                raise RuntimeError('corrupted image data')
            if self._byte_to_int(f.read(4)) != self.image_size[0]:
                raise RuntimeError('corrupted image data')
            if self._byte_to_int(f.read(4)) != self.image_size[1]:
                raise RuntimeError('corrupted image data')

            data: np.ndarray = np.frombuffer(f.read(), dtype=np.uint8)
            if data.shape[0] != self.n_data * self.image_size[0] * self.image_size[1]:
                raise RuntimeError('corrupted image data')
            self.images: np.ndarray = data.reshape(-1, self.image_size[0], self.image_size[1])

        with (folder / 'labels.bin').open('rb') as f:
            if f.read(4) != b'\x00\x00\x08\x01':
                raise RuntimeError('corrupted label data')
            if self._byte_to_int(f.read(4)) != self.n_data:
                raise RuntimeError('corrupted label data')

            self.labels: np.ndarray = np.frombuffer(f.read(), dtype=np.uint8)
            if self.labels.shape[0] != self.n_data:
                raise RuntimeError('corrupted label data')

    def __len__(self) -> int:
        return self.n_data

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self.images[index], self.labels[index]

    @staticmethod
    def _byte_to_int(x: bytes) -> int:
        return int(x.hex(), base=16)


class PartialDataset(Dataset):
    def __init__(self, source: Dataset, slice_index: list[int]) -> None:
        self.source = source
        self.slice = slice_index

    def __len__(self) -> int:
        return len(self.slice)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.source[self.slice[index]]


def random_split(dataset: Dataset, ratio: Union[float, Iterable[float]]) -> tuple[Dataset]:
    real_ratio: list[float]

    if isinstance(ratio, (int, float)):
        if ratio <= 0:
            real_ratio = [0, 1]
        elif ratio >= 1:
            real_ratio = [1, 0]
        else:
            real_ratio = [ratio, 1 - ratio]
    else:
        real_ratio = [max(x, 0) for x in ratio]
        cur_sum: float = sum(real_ratio)

        if cur_sum == 0:
            real_ratio = [1]
        else:
            real_ratio = [x / cur_sum for x in real_ratio]

    total_size: int = len(dataset)
    random_index: list[int] = list(range(total_size))
    np.random.shuffle(random_index)
    all_index_list: list[list[int]] = []
    pivot: int = 0

    for r in real_ratio:
        next_pivot: int = min(total_size, pivot + round(r * total_size))
        all_index_list.append(random_index[pivot:next_pivot])
        pivot = next_pivot

    return tuple(PartialDataset(dataset, x) for x in all_index_list)
