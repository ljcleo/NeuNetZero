from pathlib import Path

import numpy as np

from dataset import MNISTDataset, random_split
from loader import DataLoader
from logger import make_logger
from loss import SoftmaxNLL
from model import ImageClsMLP
from optimizer import SGD
from trainer import Trainer
from util import evaluate

if __name__ == '__main__':
    np.random.seed(19260817)
    root_path = Path('..')
    data_path = root_path / 'data'
    train_set, valid_set = random_split(MNISTDataset(data_path, 'training'), (0.9, 0.1))
    test_set = MNISTDataset(data_path, 'test')
    train_loader = DataLoader(train_set, 1024, True, False)
    valid_loader = DataLoader(valid_set, 1024, False, False)
    test_loader = DataLoader(test_set, 1024, False, False)
    model = ImageClsMLP(test_set.image_size, 10, [256])
    trainer = Trainer(model, SoftmaxNLL(), SGD(model, 0.0001, 0), None)
    trainer.train(train_loader, valid_loader, 200, 10, False, make_logger('test', root_path, True))

    for loader in (train_loader, valid_loader, test_loader):
        print(evaluate(model, loader))

        for i, o in loader:
            print(np.argmax(model.forward(i), axis=1)[:40])
            print(o[:40])
            break
