from typing import Callable, Tuple, Any, Iterable

import torch
from torch import nn
from torchsummary import summary  # another extension, a'la keras model.summary funtion
from torch.utils.data import DataLoader

from pegc.models import Resnet1D
from pegc import constants
from pegc.training.utils import load_full_dataset, initialize_random_seeds
from pegc.generators import PUTEEGGesturesDataset


def _validate(model: nn.Module, loss_fnc: Callable, val_gen: DataLoader, device: Any) -> Tuple[float, float]:
    model.eval()

    loss_sum = 0
    acc_sum = 0
    with torch.no_grad():
        for X_batch, y_batch in val_gen:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_pred = model(X_batch)
            loss = loss_fnc(batch_pred, y_batch)

            loss_sum += loss
            acc_sum += (y_batch.argmax(dim=1) == batch_pred.argmax(dim=1)).sum()

    return loss_sum.item() / len(val_gen.dataset), acc_sum.item() / len(val_gen.dataset)


def train_loop(dataset_dir_path: str, architecture: str, force_cpu: bool = False, epochs: int = 100,
               batch_size: int = 256, shuffle: bool = True, base_feature_maps: int = 64):
    architectures_lookup_table = {'resnet': Resnet1D}
    assert architecture in architectures_lookup_table, 'Specified model architecture unknown!'
    device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
    initialize_random_seeds(constants.RANDOM_SEED)

    model = architectures_lookup_table[architecture](constants.DATASET_FEATURES_SHAPE[0],
                                                     base_feature_maps, constants.NB_DATASET_CLASSES).to(device)
    summary(model, input_size=constants.DATASET_FEATURES_SHAPE)  # Without including batch.

    data = load_full_dataset(dataset_dir_path)
    train_dataset = PUTEEGGesturesDataset(data.X_train, data.y_train)
    val_dataset = PUTEEGGesturesDataset(data.X_test, data.y_test)
    train_gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_gen = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)  # Note: this data is quite simple, no additional workers will be required for loading/processing.
    # TODO: X_val, y_val other than from test set ; p

    callbacks = None  # TODO

    # TODO: also make adjustable? Perhaps some config file would be more handy?
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fnc = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

    # TODO/maybe: some augmentation, mixup maybe?
    for ep in range(epochs):
        model.train()
        loss_sum = 0
        acc_sum = 0

        for X_batch, y_batch in train_gen:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            batch_pred = model(X_batch)
            loss = loss_fnc(batch_pred, y_batch)

            loss_sum += loss
            acc_sum += (y_batch.argmax(dim=1) == batch_pred.argmax(dim=1)).sum()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = loss_sum.item() / len(train_gen.dataset)
        train_acc = acc_sum.item() / len(train_gen.dataset)
        val_loss, val_acc = _validate(model, loss_fnc, val_gen, device)
        print(f'Epoch {ep} train loss: {train_loss:.5f}, train acc: {train_acc:.5f}, '
              f'val loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
