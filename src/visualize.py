from collections import defaultdict
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pylab import style

from layer import Linear
from model import ImageClsMLP


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


def visualize_compare(result: dict[str, tuple[str, str, int, float, float, int, float]],
                      name: str, path: Path) -> None:
    groups: defaultdict[int, list[tuple[str, float]]] = defaultdict(list)
    group_name: list[str] = []

    for optimizer, scheduler, batch_size, _, _, _, test_acc in result.values():
        groups[batch_size].append((f'{optimizer}-{scheduler}', test_acc))

    subgroup_name: list[str] = [str(x) for x in groups]
    subgroup_name.sort()

    for v in groups.values():
        v.sort()

        if len(group_name) == 0:
            for k, _ in v:
                group_name.append(k)

    plot_data: dict[str, list[float]] = {str(k): np.array([x for _, x in v])
                                         for k, v in groups.items()}

    style.use('Solarize_Light2')
    ax: Axes = plt.figure(figsize=(15, 6)).gca()
    x_center: np.ndarray = np.arange(len(group_name))
    width: float = 0.1
    offset: float = (1 - len(subgroup_name)) / 2 * width

    for subgroup in subgroup_name:
        ax.bar(x_center + offset, plot_data[subgroup], width)
        offset += width

    ax.set_xticks(x_center, group_name)
    ax.set_xlabel('Optimizer and Scheduler')
    ax.set_ylabel('Test Accuracy')
    ax.legend([f'Batch Size {x}' for x in subgroup_name])

    plt.title(f'Model "{name}" Test Accuracy')
    plt.tight_layout()
    plt.savefig(path / f'{name}-compare.png', dpi=150)
    plt.close()
