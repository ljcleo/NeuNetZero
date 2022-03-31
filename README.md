# NeuNetZero

Neural Network & Deep Learning Homework 1

## Introduction

A two-layer MLP training and evaluation toolchain for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), written in Python 3 using NumPy.

## Requirements

- Python >= 3.9
- NumPy
- Matplotlib
- pyyaml
- requests (for automatic dataset download)

## Usage

1. Run `python download.py` to download the MNIST training and test datasets automatically (`requests` package required). Alternatively, download and extract the dataset files manually, then put them in a folder named `data` with the following structure (remember to rename the files properly):

   ```{plain}
   data
   ├── training
   │   ├── images.bin  # train-images-idx3-ubyte
   │   └── labels.bin  # train-labels-idx1-ubyte
   └── test
       ├── images.bin  # t10k-images-idx3-ubyte
       └── labels.bin  # t10k-labels-idx1-ubyte
   ```

2. Run `python main.py` to train models using the "main" configuration. To use other configuration schemes listed in the `config` folder, add `-c [config]` at the end of the command, where `[config]` is the selected configuration. All scripts listed below can select configurations with this option, the default of which is "main".

   This script will create four types of files. The training logs are stored in `log/config-name/`. The training curves along with layer visualization plots are saved in `img/config-name/`, named `model-name-curve.png` and `model-name-layer*-[start|end].png` respectively. The trained models are dumped into `model/config-name/model-name.pkl`. The summary results are written to `out/config-name.csv`, including model settings, best grid-search parameters and test accuracy of each model.

   Trained models and summaries using preset configurations can be downloaded using the following link. Just extract the Zip archive and put the folders into the root directory.

3. Run `python evalacc.py` to evaluate training and test accuracies of all models trained. Results are written to `out/config-name-eval.csv`, and a comparison bar plot is saved to `img/config-name/config-name-compare.img`.

4. Run `python confusion.py` to create confusion matrix heatmaps of all models based on the test dataset. All heatmaps are saved to `img/config-name/model-name-confusion.png`.

5. Run `python errorcase.py` to plot error case samples of all models chosen from the test dataset. Use the `-n [num-cases]` option to control the number of error cases. Results are saved to `img/config-name/model-name-errorcase.png`.

## Configuration Format

```{yaml}
name: mlp2                # Model name
dropout_rate: 0           # Dropout rate (set to 0 to turn off)
shuffle: true             # Shuffle dataset per epoch
drop_last: false          # Drop last batch if smaller than batch size
max_epoch: 10             # Maximum training epochs
patience: 5               # Early stopping patience (epoch)
improve_threshold: 0.001  # Early stopping relative improvement threshold
use_best_param: true      # Use recorded best parameters after training
compare:
  optimizer:              # Candidate optimizers (SGD, Momentum SGD)
    - name: SGD
    - name: MomentumSGD
      params:             # Optimizer parameters if necessary
        beta: 0.9
  scheduler:              # Candidate schedulers (none, milestone, exponential decay)
    - name: null
    - name: MilestoneLR
      params:             # Scheduler parameters if necessary
        milestones:
          - 20
          - 50
          - 100
        multipliers:
          - 0.5
          - 0.1
          - 0.01
    - name: ExpDecayLR
      params:
        init: 1
        alpha: 0.96
  batch_size:             # Candidate batch sizes
    - 64
    - 256
grid_search:
  learning_rate:          # Grid search learning rates
    - 0.0002
    - 0.001
  l2_lambda:              # Grid search L2 regularization weights
    - 0
    - 0.001
  hidden_size:            # Grid search hidden layer sizes
    - 256
    - 768
```

## Project Structure

```{plain}
NeuNetZero
├── config
│   ├── dropout.yaml        # Grid search multiple model with dropout
│   ├── main.yaml           # Grid search multiple model without dropout
│   ├── single.yaml         # Single best model without dropout
│   ├── sindrop.yaml        # Single best model with dropout
│   └── test.yaml           # Functionality test
├── data
│   ├── training
│   │   ├── images.bin      # Training dataset images
│   │   └── labels.bin      # Training dataset labels
│   └── test
│       ├── images.bin      # Test dataset images
│       └── labels.bin      # Test dataset labels
├── img                     # Plots
├── log                     # Logs
├── model                   # Dumped models
├── out                     # Summary results
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataset.py      # Generic and MNIST dataset class
│   │   └── loader.py       # Data loader class
│   ├── nn
│   │   ├── __init__.py
│   │   ├── initializer.py  # Layer parameter initializers
│   │   ├── layer.py        # Neural network layers
│   │   ├── loss.py         # Loss functions
│   │   ├── model.py        # Generic and Image classification MLP model
│   │   ├── optimizer.py    # Optimizers
│   │   └── scheduler.py    # Learning rate decay schedulers
│   ├── tool
│   │   ├── __init__.py
│   │   ├── logger.py       # Logger creator
│   │   ├── util.py         # Utility functions
│   │   └── visualize.py    # Visualization methods
│   └── train
│       ├── __init__.py
│       ├── task.py         # Grid search and model setting comparison methods
│       └── trainer.py      # Trainer class
├── confusion.py            # Confusion matrix script
├── download.py             # Dataset download script
├── errorcase.py            # Error case script
├── evalacc.py              # Accuracy evaluation script
├── main.py                 # Model training script
└── readme.md               # This file
```

## Author

Jingcong Liang, [18307110286](mailto:18307110286@fudan.edu.cn)
