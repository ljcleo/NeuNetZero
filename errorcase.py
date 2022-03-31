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
from src.tool.visualize import visualize_error_case

if __name__ == '__main__':
    np.random.seed(19260817)
    root_path: Path = Path('.')

    parser: ArgumentParser = ArgumentParser(
        description='visualize MNIST MLP classifier error cases'
    )

    parser.add_argument('-c', '--config', default='main', help='model config')
    parser.add_argument('-n', '--num-cases', default=9, type=int,
                        help='maximum number of error cases')

    args: Namespace = parser.parse_args()
    config_name: str = args.config
    n_cases: int = args.num_cases

    logger: Logger = make_logger(f'{config_name}-errorcase', root_path, True)
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

                logger.info(f'Visualizing error cases of {real_name} ...')
                test_loader: DataLoader = DataLoader(test_set, batch_size, config['shuffle'],
                                                     config['drop_last'])

                model: ImageClsMLP = ImageClsMLP(test_set.image_size, 10, [best_hidden[real_name]],
                                                 config['dropout_rate'])
                model.load_params(model_path)
                model.toggle_train(False)

                error_input: list[np.ndarray] = []
                error_output: list[np.ndarray] = []
                error_pred: list[np.ndarray] = []
                n_current: int = 0

                for input_batch, output_batch in test_loader:
                    pred: np.ndarray = np.argmax(model.forward(input_batch), axis=1)
                    index: np.ndarray = np.flatnonzero(pred != output_batch)[:n_cases - n_current]
                    error_input.append(input_batch[index])
                    error_output.append(output_batch[index])
                    error_pred.append(pred[index])
                    n_current += index.shape[0]

                    if n_current >= n_cases:
                        break

                visualize_error_case(np.concatenate(error_input), np.concatenate(error_output),
                                     np.concatenate(error_pred), 3, real_name,
                                     get_path(root_path, 'img', name))

    logger.info('Finished visualizing error cases of all models.')
