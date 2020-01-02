import os.path as osp
from collections import namedtuple

import numpy as np


PUTEEGDataset = namedtuple('PUTEEGDATASET', ('X_train', 'y_train', 'X_test', 'y_test'))


def load_full_dataset(dataset_dir_path: str) -> PUTEEGDataset:
    X_train = np.load(osp.join(dataset_dir_path, 'X_train.npy'))
    y_train = np.load(osp.join(dataset_dir_path, 'y_train.npy'))
    X_test = np.load(osp.join(dataset_dir_path, 'X_test.npy'))
    y_test = np.load(osp.join(dataset_dir_path, 'y_test.npy'))
    # swap axes as pytorch is channels first
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    dataset = PUTEEGDataset(X_train, y_train, X_test, y_test)
    return dataset
