import numpy as np

from torch.utils.data import Dataset


class PUTEEGGesturesDataset(Dataset):

    def __init__(self, X: np.array, y: np.array):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
