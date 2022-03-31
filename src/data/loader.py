from typing import Any, Iterator

import numpy as np

from .dataset import Dataset


class DataIterator:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool):
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.seq: list[int] = list(range(len(dataset)))
        self.pivot: int = 0

        if shuffle:
            np.random.shuffle(self.seq)

        if drop_last:
            while len(self.seq) % batch_size > 0:
                self.seq.pop()

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.pivot >= len(self.seq):
            raise StopIteration()

        next_pivot: int = self.pivot + self.batch_size
        current: list[Any] = [self.dataset[i] for i in self.seq[self.pivot:next_pivot]]
        self.pivot = next_pivot
        return tuple(np.stack(x) for x in zip(*current))


class DataLoader:
    def __init__(self, data: Dataset, batch_size: int, shuffle: bool, drop_last: bool):
        self.data: Dataset = data
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.drop_last: bool = drop_last

    def __iter__(self) -> DataIterator:
        return DataIterator(self.data, self.batch_size, self.shuffle, self.drop_last)
