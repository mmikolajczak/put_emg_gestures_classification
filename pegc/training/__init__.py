from typing import Callable, Tuple, Any, Dict
import os
import os.path as osp

import torch
from torch import nn
from torchsummary import summary  # another extension, a'la keras model.summary funtion
from torch.utils.data import DataLoader

from pegc.models import Resnet1D
from pegc import constants
from pegc.training.utils import load_full_dataset, initialize_random_seeds, mixup_batch, AverageMeter
from pegc.generators import PUTEEGGesturesDataset


def _validate(model: nn.Module, loss_fnc: Callable, val_gen: DataLoader, device: Any) -> Tuple[float, float]:
    model.eval()

    loss_tracker = AverageMeter()
    acc_tracker = AverageMeter()
    with torch.no_grad():
        for X_batch, y_batch in val_gen:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_pred = model(X_batch)
            loss = loss_fnc(batch_pred, y_batch)

            loss_tracker.update(loss.item(), len(batch_pred))
            acc_tracker.update((y_batch.argmax(dim=1) == batch_pred.argmax(dim=1)).sum().item() / len(batch_pred))

    return loss_tracker.avg, acc_tracker.avg


def _epoch_train(model: nn.Module, train_gen: DataLoader, device: Any, optimizer: Any, loss_fnc: Callable,
                 epoch_idx: int, use_mixup: bool, alpha: float) -> Dict[str, float]:
    model.train()
    loss_tracker = AverageMeter()

    for batch_idx, (X_batch, y_batch) in enumerate(train_gen, start=1):
        if use_mixup:  # Problem: acc will stop being meaningful for training due to that (mse/mae instead?)
            X_batch, y_batch = mixup_batch(X_batch, y_batch, alpha)

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        batch_y_pred = model(X_batch)
        loss = loss_fnc(batch_y_pred, y_batch)
        loss_tracker.update(loss.item(), len(batch_y_pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'\rEpoch {epoch_idx} [{batch_idx}/{len(train_gen)}]: '
              f'Loss: {loss_tracker.val:.4f} (mean: {loss_tracker.avg:.4f})', end='')

    return {'loss': loss_tracker.avg}


def train_loop(dataset_dir_path: str, architecture: str, results_dir_path: str, force_cpu: bool = False,
               epochs: int = 100, batch_size: int = 256, shuffle: bool = True, base_feature_maps: int = 64,
               use_mixup=True, alpha: float = 1) -> None:
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

    epochs = 10
    os.makedirs(results_dir_path, exist_ok=True)
    for ep in range(1, epochs + 1):
        epoch_stats = _epoch_train(model, train_gen, device, optimizer, loss_fnc, ep, use_mixup, alpha)

        val_loss, val_acc = _validate(model, loss_fnc, val_gen, device)
        print(f'\nEpoch {ep} train loss: {epoch_stats["loss"]:.4f}, '
              f'val loss: {val_loss:.5f}, val_acc: {val_acc:.4f}')
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'last_val_loss': val_loss,
                'last_val_acc': val_acc,
                'epoch': ep}, osp.join(results_dir_path, 'final_checkpoint.tar'))
