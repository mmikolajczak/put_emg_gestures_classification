import os
import os.path as osp
from collections import namedtuple
from typing import Tuple, Dict, Any

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


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        # self.reset() would be sufficient but since lint is complaining about definitions outside of init... ; p

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, samples: int = 1):
        self.val = value
        self.sum += value * samples
        self.count += samples
        self.avg = self.sum / self.count

    # Note/TODO: add smoothing perhaps.


class EarlyStoppingSignal(RuntimeError):

    def __init__(self, message):
        super(EarlyStoppingSignal, self).__init__(message)


class EarlyStopping:

    def __init__(self, monitor: str, mode: str = 'min', min_delta: float = 0, patience: int = 10,
                 percentage: bool = False):
        assert mode in ('min', 'max')
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def on_epoch_end(self, epoch: int, epoch_stats: dict):
        metric = epoch_stats.get(self.monitor)
        if self.best is None:
            self.best = metric

        if np.isnan(metric):
            raise EarlyStoppingSignal(f'Training early stopped due to lack of improvement from {self.patience} epochs!')

        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            raise EarlyStoppingSignal(f'Training early stopped due to lack of improvement from {self.patience} epochs!')

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class ModelCheckpoint:

    def __init__(self, save_dir: str, monitor: str, checkpointed: Dict[str, Any],
                 save_best_only: bool = False, verbose: int = 0):
        if save_dir.startswith('~'):
            save_dir = osp.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.monitor = monitor
        self.checkpointed = checkpointed
        # Mode = 'min' for best only supported – at least as of now, later can be adapted from the class above.
        self.best_loss = float('inf')

    def save_checkpoint(self, epoch: int, file: str):
        to_save = {f'{k}_state_dict': v.state_dict() for k, v in self.checkpointed.items()}
        to_save['epoch'] = epoch
        torch.save(to_save, osp.join(self.save_dir, file))

    def on_epoch_end(self, epoch: int, epoch_stats: dict):
        filename = f'ep_{epoch}_checkpoint.tar'
        if self.save_best_only:
            current_loss = epoch_stats.get(self.monitor)
            if current_loss < self.best_loss:
                if self.verbose > 0:
                    print(f'Epoch {epoch}: improved from {self.best_loss:.4f} to {current_loss:.4f} '
                          f'saving model to {filename}')
                self.best_loss = current_loss
                self.save_checkpoint(epoch, filename)
        else:
            if self.verbose > 0:
                print(f'Epoch {epoch}: saving model to ep_{epoch}_checkpoint.tar')
            self.save_checkpoint(epoch, filename)
