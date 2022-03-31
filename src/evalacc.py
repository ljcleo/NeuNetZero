from argparse import ArgumentParser, Namespace
from csv import DictReader, writer
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
from yaml import safe_load

from dataset import MNISTDataset
from loader import DataLoader
from logger import make_logger
from model import ImageClsMLP
from util import get_path, evaluate
from visualize import visualize_compare

if __name__ == '__main__':
    np.random.seed(19260817)
    root_path: Path = Path('..')

    parser: ArgumentParser = ArgumentParser(description='evaluate MNIST MLP classifier accuracy')
    parser.add_argument('-c', '--config', default='main', help='model config')
    args: Namespace = parser.parse_args()
    config_name: str = args.config

    logger: Logger = make_logger(f'{config_name}-test', root_path, True)
    logger.info(f'Loading model configuration from "{config_name}.yaml" ...')

    with (get_path(root_path, 'config') / f'{config_name}.yaml').open('r', encoding='utf8') as f:
        config: dict[str, Any] = safe_load(f)

    name: str = f'{config_name}-{config["name"]}'
    compare_config: dict[str, Any] = config['compare']

    best_hidden: dict[str, int] = {}
    logger.info(f'Loading hyper-parameters from "{config_name}.csv" ...')

    with (get_path(root_path, 'out') / f'{config_name}.csv').open('r', encoding='utf8') as f:
        reader: DictReader = DictReader(f)
        for row in reader:
            best_hidden[row['name']] = int(row['best_hidden'])

    data_path: Path = get_path(root_path, 'data')
    training_set: MNISTDataset = MNISTDataset(data_path, 'training')
    test_set: MNISTDataset = MNISTDataset(data_path, 'test')

    visualize_result: list[tuple[str, str, int, float]] = []
    csv_result: list[tuple[str, float, float]] = []

    for i, optimizer in enumerate(compare_config['optimizer']):
        for j, scheduler in enumerate(compare_config['scheduler']):
            for k, batch_size in enumerate(compare_config['batch_size']):
                real_name: str = f'{name}-{i}{j}{k}'
                model_path: Path = get_path(root_path, 'model', name) / f'{real_name}.pkl'

                if not model_path.exists():
                    continue
                logger.info(f'Testing accuracy of {real_name} ...')

                model: ImageClsMLP = ImageClsMLP(test_set.image_size, 10, [best_hidden[real_name]],
                                                 config['dropout_rate'])
                model.load_params(model_path)
                model.toggle_train(False)

                train_acc: float = evaluate(model, DataLoader(
                    training_set, batch_size, config['shuffle'],  config['drop_last']
                ))
                test_acc: float = evaluate(model, DataLoader(
                    test_set, batch_size, config['shuffle'], config['drop_last']
                ))

                visualize_result.append((optimizer['name'], scheduler['name'],
                                         batch_size, test_acc))
                csv_result.append((real_name, train_acc, test_acc))

    visualize_compare(visualize_result, config_name, get_path(root_path, 'img', name))

    with (get_path(root_path, 'out') / f'{config_name}-test.csv').open('w', encoding='utf8',
                                                                       newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(('name', 'train_acc', 'test_acc'))
        csv_writer.writerows(csv_result)

    logger.info('Finished evaluating accuracy of all models.')
