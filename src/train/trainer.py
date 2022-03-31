from logging import Logger
from time import time
from typing import Optional, Type

import numpy as np

from ..data.loader import DataLoader
from ..nn.loss import Loss
from ..nn.model import Model
from ..nn.optimizer import Optimizer
from ..nn.scheduler import Scheduler


class Trainer:
    def __init__(self, model: Model, loss: Type[Loss], optimizer: Optimizer,
                 scheduler: Optional[Scheduler]) -> None:
        self.model: Model = model
        self.loss: Type[Loss] = loss
        self.optimizer: Optimizer = optimizer
        self.scheduler: Optional[Scheduler] = scheduler

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, max_epoch: int,
              patience: int, improve_threshold: float, use_best_param: bool,
              logger: Logger) -> tuple[list[float], list[float], list[float]]:
        best_loss: float = self._evaluate_loss_acc(valid_loader)[0]
        best_params: np.ndarray = self.model.get_params_and_grads()[0]

        train_loss: list[float] = []
        valid_loss: list[float] = []
        valid_acc: list[float] = []

        cur_patience: int = 0
        train_start_time: float = time()
        logger.info('Start training model ...')

        try:
            for cur_epoch in range(max_epoch):
                batch_start_time: float = time()
                cur_train_loss: float = 0
                cur_valid_loss: float = 0
                cur_valid_acc: float = 0
                data_len: int = 0

                if self.scheduler is not None:
                    self.optimizer.set_multiplier(self.scheduler.calc_multiplier(cur_epoch))

                for input_batch, output_batch in train_loader:
                    cur_len: int = output_batch.shape[0]
                    data_len += cur_len
                    cur_train_loss += self._train_iter(input_batch, output_batch) * cur_len

                if self.model.check_nan():
                    raise ValueError('found inf or nan among model parameters')

                cur_train_loss /= data_len
                cur_valid_loss, cur_valid_acc = self._evaluate_loss_acc(valid_loader)
                train_loss.append(cur_train_loss)
                valid_loss.append(cur_valid_loss)
                valid_acc.append(cur_valid_acc)

                logger.info(f'Epoch: {cur_epoch + 1} Training Loss: {cur_train_loss:.4f} ' +
                            f'Validation Loss: {cur_valid_loss:.4f} ' +
                            f'Validation Accuracy: {cur_valid_acc:.4f} ' +
                            f'Elapsed Time: {time() - batch_start_time:.2f}s')

                if (best_loss - cur_valid_loss) / best_loss > improve_threshold:
                    best_loss = cur_valid_loss
                    best_params = self.model.get_params_and_grads()[0]
                    cur_patience = 0
                else:
                    cur_patience += 1

                if cur_patience == patience:
                    break
        except ValueError as e:
            logger.warn(e, exc_info=False)

        if use_best_param:
            self.model.set_params(best_params)

        logger.info(f'Finished training model. Elapsed time: {time() - train_start_time:.2f}s')
        return train_loss, valid_loss, valid_acc

    def _train_iter(self, input_batch: np.ndarray, output_batch: np.ndarray) -> float:
        self.model.toggle_train(True)
        pred: np.ndarray = self.model.forward(input_batch)
        loss_grad: np.ndarray = self.loss.gradient(pred, output_batch)
        self.model.backward(loss_grad)
        self.optimizer.update()
        return self.loss.loss(pred, output_batch)

    def _evaluate_loss_acc(self, data: DataLoader) -> tuple[float, float]:
        self.model.toggle_train(False)
        all_pred: list[np.ndarray] = []
        all_label: list[np.ndarray] = []

        for input_batch, output_batch in data:
            all_pred.append(self.model.forward(input_batch))
            all_label.append(output_batch)

        pred: np.ndarray = np.concatenate(all_pred)
        label: np.ndarray = np.concatenate(all_label)
        return self.loss.loss(pred, label), (np.argmax(pred, axis=1) == label).mean()
