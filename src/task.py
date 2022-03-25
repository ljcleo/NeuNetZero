from logging import Logger
from pathlib import Path
from time import time
from typing import Any, Optional, Type

import numpy as np

from dataset import Dataset
from loader import DataLoader
from logger import make_logger
from loss import SoftmaxNLL
from model import ImageClsMLP
from optimizer import SGD, MomentumSGD, Optimizer
from scheduler import ExpDecayLR, MilestoneLR, Scheduler
from trainer import Trainer
from util import evaluate, get_path
from visualize import visualize_params, visualize_training

optimizer_dict: dict[str, Type[Optimizer]] = {'SGD': SGD, 'MomentumSGD': MomentumSGD}
scheduler_dict: dict[str, Type[Scheduler]] = {'ExpDecayLR': ExpDecayLR, 'MilestoneLR': MilestoneLR}


def mnist_compare(image_size: tuple[int, int], n_class: int, learning_rate: list[float],
                  l2_lambda: list[float], hidden_size: list[int],
                  optimizer: list[dict[str, Any]], scheduler: list[Optional[dict[str, Any]]],
                  batch_size: list[int], train_set: Dataset, valid_set: Dataset, test_set: Dataset,
                  shuffle: bool, drop_last: bool, max_epoch: int, patience: int,
                  use_best_param: bool, name: str,
                  root_path: Path) -> dict[str, tuple[str, str, int, float, float, int, float]]:
    result: dict[str, tuple[str, str, int, float, float, int, float]] = {}
    start_time: float = time()
    logger: Logger = make_logger(name, root_path, True)
    logger.info('Start training models ...')

    for i, opt in enumerate(optimizer):
        for j, sch in enumerate(scheduler):
            for k, bs in enumerate(batch_size):
                search_start_time: float = time()
                new_name: str = f'{name}-{i}{j}{k}'
                train_loader: DataLoader = DataLoader(train_set, bs, shuffle, drop_last)
                valid_loader: DataLoader = DataLoader(valid_set, bs, False, False)
                test_loader: DataLoader = DataLoader(test_set, bs, False, False)

                best_hyper_params: tuple[float, float, int] = mnist_grid_search(
                    image_size, n_class, learning_rate, l2_lambda, hidden_size,
                    optimizer_dict[opt['name']], opt.get('params', {}),
                    None if sch['name'] is None else
                    scheduler_dict[sch['name']](**sch.get('params', {})),
                    train_loader, valid_loader, max_epoch, patience, use_best_param, new_name,
                    root_path
                )

                best_model: ImageClsMLP = ImageClsMLP(image_size, n_class, [best_hyper_params[2]])
                best_model.load_params(get_path(root_path, 'model', name) / f'{new_name}.pkl')
                test_acc: float = evaluate(best_model, test_loader)
                result[new_name] = (opt['name'], sch['name'], bs) + best_hyper_params + (test_acc, )

                logger.info(f'Finished grid search. Test Accuracy: {test_acc:.4f} ' +
                            f'Elapsed Time: {time() - search_start_time:.2f}s\n' +
                            f'Optimizer: {opt["name"]}\n' +
                            f'Scheduler: {sch["name"]}\n' +
                            f'Batch Size: {bs}\n' +
                            f'Best Learning Rate: {best_hyper_params[0]}\n' +
                            f'Best L2 Lambda: {best_hyper_params[1]}\n' +
                            f'Best Hidden Layer Size: {best_hyper_params[2]}')

    logger.info(f'Finished training. Elapsed Time: {time() - start_time:.2f}s')
    return result


def mnist_grid_search(image_size: tuple[int, int], n_class: int, learning_rate: list[float],
                      l2_lambda: list[float], hidden_size: list[int],
                      optimizer_type: Type[Optimizer], optimizer_params: dict[str, Any],
                      scheduler: Optional[Scheduler], train_loader: DataLoader,
                      valid_loader: DataLoader, max_epoch: int, patience: int, use_best_param: bool,
                      name: str, root_path: Path) -> tuple[float, float, int]:
    best_hyper_params: Optional[tuple[float, float, int]] = None
    best_params: Optional[tuple[np.ndarray, np.ndarray]] = None
    best_records: Optional[tuple[list[float], list[float], list[float]]] = None
    best_valid_acc: Optional[float] = None
    search_start_time: float = time()

    logger: Logger = make_logger(name, root_path, True)
    logger.info('Start grid searching ...')

    for i, lr in enumerate(learning_rate):
        for j, l2 in enumerate(l2_lambda):
            for k, hidden in enumerate(hidden_size):
                train_start_time: float = time()
                model: ImageClsMLP = ImageClsMLP(image_size, n_class, [hidden])
                old_params: np.ndarray = model.get_params_and_grads()[0]
                optimizer: Optimizer = optimizer_type(model=model, lr=lr, l2_lambda=l2,
                                                      **optimizer_params)

                trainer: Trainer = Trainer(model, SoftmaxNLL(), optimizer, scheduler)
                records: tuple[list[float], list[float], list[float]] = trainer.train(
                    train_loader, valid_loader, max_epoch, patience, use_best_param,
                    make_logger(f'{name}-{i}{j}{k}', root_path, False)
                )

                valid_acc: float = evaluate(model, valid_loader)

                if best_valid_acc is None or valid_acc > best_valid_acc:
                    best_hyper_params = lr, l2, hidden
                    best_params = old_params, model.get_params_and_grads()[0]
                    best_records = records
                    best_valid_acc = valid_acc

                logger.info(f'Learning Rate: {lr} L2 Lambda: {l2} Hidden Layer Size: {hidden} ' +
                            f'Elapsed Time: {time() - train_start_time:.2f}s')

    img_path: Path = get_path(root_path, 'img', name)
    visualize_training(best_records[0], best_records[1], best_records[2], name, img_path)
    final_model: ImageClsMLP = ImageClsMLP(image_size, n_class, [best_hyper_params[2]])
    final_model.set_params(best_params[0])
    visualize_params(final_model, name, 'start', img_path)
    final_model.set_params(best_params[1])
    visualize_params(final_model, name, 'end', img_path)
    final_model.dump_params(get_path(root_path, 'model', name) / f'{name}.pkl')

    logger.info(f'Finished grid search. Elapsed Time: {time() - search_start_time:.2f}s\n' +
                f'Best Learning Rate: {best_hyper_params[0]}\n' +
                f'Best L2 Lambda: {best_hyper_params[1]}\n' +
                f'Best Hidden Layer Size: {best_hyper_params[2]}')

    return best_hyper_params
