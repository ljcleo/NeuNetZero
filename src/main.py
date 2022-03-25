from argparse import ArgumentParser
from csv import writer
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
from yaml import safe_load

from dataset import MNISTDataset, PartialDataset, random_split
from logger import make_logger
from util import mnist_compare
from visualize import visualize_compare

if __name__ == '__main__':
    np.random.seed(19260817)
    root_path: Path = Path('..')

    for subfolder in ('log', 'img', 'model', 'out'):
        (root_path / subfolder).mkdir(exist_ok=True)

    parser: ArgumentParser = ArgumentParser(description='MNIST MLP classifier trainer')
    parser.add_argument('-c', '--config', default='main', help='training config')
    config_name: str = parser.parse_args().config

    logger: Logger = make_logger(config_name, root_path, True)
    logger.info(f'Loading model & training configuration from "{config_name}.yaml" ...')

    with (root_path / 'config' / f'{config_name}.yaml').open('r', encoding='utf8') as f:
        config: dict[str, Any] = safe_load(f)
    name: str = f'{config_name}-{config["name"]}'

    train_set: PartialDataset
    valid_set: PartialDataset
    train_set, valid_set = random_split(MNISTDataset(root_path / 'data', 'training'), (0.9, 0.1))
    test_set: MNISTDataset = MNISTDataset(root_path / 'data', 'test')

    compare_config: dict[str, Any] = config['compare']
    grid_search_config: dict[str, Any] = config['grid_search']
    logger.info('Start training ...')

    compare_result: dict[str, tuple[str, str, int, float, float, int, float]] = mnist_compare(
        test_set.image_size, 10, grid_search_config['learning_rate'],
        grid_search_config['l2_lambda'], grid_search_config['hidden_size'],
        compare_config['optimizer'], compare_config['scheduler'], compare_config['batch_size'],
        train_set, valid_set, test_set, config['shuffle'], config['drop_last'], config['max_epoch'],
        config['patience'], config['use_best_param'], name, root_path
    )

    logger.info('Finished training. Writing results ...')
    visualize_compare(compare_result, config_name, root_path / 'img')

    with (root_path / 'out' / f'{config_name}.csv').open('w', encoding='utf8', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(('name', 'optimizer', 'scheduler', 'batch_size',
                             'best_lr', 'best_l2', 'best_hidden', 'test_acc'))

        for name, result in compare_result.items():
            csv_writer.writerow((name, ) + result)

    logger.info('Finished writting results.')
