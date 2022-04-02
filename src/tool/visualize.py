from itertools import cycle
from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pylab import style

from ..nn.layer import Linear
from ..nn.model import ImageClsMLP


def visualize_params(model: ImageClsMLP, name: str, suffix: str, path: Path) -> None:
    style.use('Solarize_Light2')
    layer_count: int = 0

    for layer in model.modules:
        if isinstance(layer, Linear):
            layer_count += 1
            plt.figure(figsize=(6, 6))
            plt.matshow(layer.get_params_and_grads()[0].reshape(-1, layer.bias.shape[1]), 0,
                        cmap='PRGn', aspect='auto', vmin=-0.5, vmax=0.5)

            plt.colorbar()
            plt.title(f'Model "{name}" Linear Layer {layer_count} ({suffix.capitalize()})')
            plt.tight_layout()
            plt.savefig(path / f'{name}-layer{layer_count}-{suffix}.png', dpi=150)
            plt.close()


def visualize_training(train_loss: list[float], valid_loss: list[float], valid_acc: list[float],
                       name: str, path: Path) -> None:
    style.use('Solarize_Light2')
    colors: cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ax1: Axes = plt.figure(figsize=(6, 6)).gca()
    ax1.plot(train_loss, color=next(colors), label='Training Loss')
    ax1.plot(valid_loss, color=next(colors), label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.semilogy()

    ax2: Axes = ax1.twinx()
    ax2.plot(valid_acc, color=next(colors), label='Validation Accuracy')
    ax2.set_ylabel('Accuracy')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')

    plt.title(f'Model "{name}" Training Curve')
    plt.tight_layout()
    plt.savefig(path / f'{name}-curve.png', dpi=150)
    plt.close()


def visualize_compare(result: list[tuple[str, str, int, float]], name: str, path: Path) -> None:
    group_name: list[str] = []
    subgroup_name: list[str] = []
    convert: dict[tuple[str, str], float] = {}
    min_acc: float = 1

    for optimizer, scheduler, batch_size, test_acc in result:
        group: str = f'{optimizer}-{scheduler}'
        subgroup: str = f'Batch Size {batch_size}'
        convert[group, subgroup] = test_acc
        min_acc = min(min_acc, test_acc)

        if group not in group_name:
            group_name.append(group)
        if subgroup not in subgroup_name:
            subgroup_name.append(subgroup)

    plot_data: dict[str, list[float]] = {
        subgroup: np.array([convert[group, subgroup] for group in group_name])
        for subgroup in subgroup_name
    }

    style.use('Solarize_Light2')
    ax: Axes = plt.figure(figsize=(max(6, round(len(group_name) * 2.5)), 6)).gca()
    x_center: np.ndarray = np.arange(len(group_name))
    width: float = 0.6 / len(subgroup_name)
    offset: float = (1 - len(subgroup_name)) / 2 * width

    for subgroup in subgroup_name:
        ax.bar(x_center + offset, plot_data[subgroup], width)
        offset += width

    ax.set_xticks(x_center, group_name)
    ax.set_xlabel('Optimizer-Scheduler')
    ax.set_ylim([floor(min_acc * 10) / 10, 1])
    ax.set_ylabel('Test Accuracy')
    ax.legend(subgroup_name)

    plt.title(f'Model "{name}" Test Accuracy')
    plt.tight_layout()
    plt.savefig(path / f'{name}-compare.png', dpi=150)
    plt.close()


def visualize_error_case(input_batch: np.ndarray, output_batch: np.ndarray, prob: np.ndarray,
                         columns: int, name: str, path: Path) -> None:
    batch_size: int = input_batch.shape[0]
    rows: int = (batch_size - 1) // columns + 1

    style.use('Solarize_Light2')
    plt.figure(figsize=(2 * columns, 2 * rows))

    for i in range(batch_size):
        pred: int = np.argmax(prob[i])
        label: int = output_batch[i]

        plt.subplot(rows, columns, i + 1)
        plt.imshow(input_batch[i], cmap='binary')
        plt.axis('off')
        plt.title(f'{pred}: {prob[i, pred]:.0%} | {label}: {prob[i, label]:.0%}',
                  fontdict={'fontsize': 12})

    plt.suptitle(f'Model "{name}" Error Case', size=20)
    plt.tight_layout()
    plt.savefig(path / f'{name}-errorcase.png', dpi=150)
    plt.close()


def visualize_confusion_matrix(n_class: int, pred: np.ndarray, label: np.ndarray,
                               name: str, path: Path) -> None:
    matrix: np.ndarray = np.zeros((n_class, n_class), dtype=np.int32)
    for coord in zip(pred, label):
        matrix[coord] += 1

    style.use('Solarize_Light2')
    plt.figure(figsize=(6, 6))
    plt.matshow(matrix, 0, cmap='PRGn')
    plt.colorbar(shrink=0.75)

    for i in range(n_class):
        for j in range(n_class):
            plt.text(i, j, matrix[j, i], c='white', va='center', ha='center')

    plt.xticks(np.arange(10), np.arange(10))
    plt.xlabel('Label')
    plt.yticks(np.arange(10), np.arange(10))
    plt.ylabel('Prediction')
    plt.grid(False)
    plt.title(f'Model "{name}" Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path / f'{name}-confusion.png', dpi=150)
    plt.close()
