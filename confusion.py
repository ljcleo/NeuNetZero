from argparse import ArgumentParser, Namespace
from csv import DictReader
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
from yaml import safe_load

from src.data.dataset import MNISTDataset
from src.data.loader import DataLoader
from src.nn.model import ImageClsMLP
from src.tool.logger import make_logger
from src.tool.util import get_path
from src.tool.visualize import visualize_confusion_matrix

if __name__ == '__main__':
    np.random.seed(19260817)
    root_path: Path = Path('.')

    parser: ArgumentParser = ArgumentParser(
        description='visualize MNIST MLP classifier test confusion matrix'
    )

    parser.add_argument('-c', '--config', default='main', help='model config')
    args: Namespace = parser.parse_args()
    config_name: str = args.config

    logger: Logger = make_logger(f'{config_name}-confusion', root_path, True)
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

    test_set: MNISTDataset = MNISTDataset(get_path(root_path, 'data'), 'test')

    for i in range(len(compare_config['optimizer'])):
        for j in range(len(compare_config['scheduler'])):
            for k, batch_size in enumerate(compare_config['batch_size']):
                real_name: str = f'{name}-{i}{j}{k}'
                model_path: Path = get_path(root_path, 'model', name) / f'{real_name}.pkl'

                if not model_path.exists():
                    continue

                logger.info(f'Visualizing confusion matrix of {real_name} ...')
                test_loader: DataLoader = DataLoader(test_set, batch_size, config['shuffle'],
                                                     config['drop_last'])

                model: ImageClsMLP = ImageClsMLP(test_set.image_size, 10, [best_hidden[real_name]],
                                                 config['dropout_rate'])
                model.load_params(model_path)
                model.toggle_train(False)
                all_pred: list[np.ndarray] = []
                all_label: list[np.ndarray] = []

                for input_batch, output_batch in test_loader:
                    all_pred.append(model.forward(input_batch))
                    all_label.append(output_batch)

                visualize_confusion_matrix(10, np.argmax(np.concatenate(all_pred), axis=1),
                                           np.concatenate(all_label), real_name,
                                           get_path(root_path, 'img', name))

    logger.info('Finished visualizing confusion matrix of all models.')
