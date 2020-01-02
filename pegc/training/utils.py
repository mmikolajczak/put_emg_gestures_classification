import os.path as osp
from collections import namedtuple
from typing import Tuple

import numpy as np
import torch


PUTEEGDataset = namedtuple('PUTEEGDATASET', ('X_train', 'y_train', 'X_test', 'y_test'))


def load_full_dataset(dataset_dir_path: str) -> PUTEEGDataset:
    X_train = np.load(osp.join(dataset_dir_path, 'X_train.npy')).astype(np.float32)
    y_train = np.load(osp.join(dataset_dir_path, 'y_train.npy')).astype(np.float32)
    X_test = np.load(osp.join(dataset_dir_path, 'X_test.npy')).astype(np.float32)
    y_test = np.load(osp.join(dataset_dir_path, 'y_test.npy')).astype(np.float32)
    # swap axes as pytorch is channels first
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    dataset = PUTEEGDataset(X_train, y_train, X_test, y_test)
    return dataset


def initialize_random_seeds(seed: int) -> None:
    # Note: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits or
    # different platforms. Furthermore, results need not be reproducible between CPU and GPU executions, even
    # when using identical seeds." But using the code below, the runs for specific platform/release should've been
    # made deterministic (welp, at least according to docs).
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # Note: type hints are for torch.Tensor but it is rather universal and will also work if input is just numpy.
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = len(X)
    index = torch.randperm(batch_size)

    mixed_X = lam * X + (1 - lam) * X[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_X, mixed_y
