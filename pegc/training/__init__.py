from typing import Callable, Any, Dict, Iterable
import os
import os.path as osp

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from torch import nn
from torchsummary import summary  # another extension, a'la keras model.summary funtion
from torch.utils.data import DataLoader

from pegc.models import Resnet1D
from pegc import constants
from pegc.training.utils import load_full_dataset, initialize_random_seeds, mixup_batch, AverageMeter, \
    EarlyStopping, EarlyStoppingSignal, ModelCheckpoint, save_json
from pegc.training.clr import CyclicLR
from pegc.training.radam import RAdam
from pegc.training.lookahead import Lookahead
from pegc.generators import PUTEEGGesturesDataset


def _validate(model: nn.Module, loss_fnc: Callable, val_gen: DataLoader, device: Any) -> Dict[str, float]:
    model.eval()

    loss_tracker = AverageMeter()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_gen:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_pred = model(X_batch)
            loss = loss_fnc(batch_pred, y_batch)

            loss_tracker.update(loss.item(), len(batch_pred))
            y_true.append(np.argmax(y_batch.cpu().numpy(), axis=1))
            y_pred.append(np.argmax(batch_pred.cpu().numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {'val_loss': loss_tracker.avg, 'val_acc': acc, 'cm': cm.tolist()}


def _epoch_train(model: nn.Module, train_gen: DataLoader, device: Any, optimizer: Any, loss_fnc: Callable,
                 epoch_idx: int, use_mixup: bool, alpha: float, schedulers: Iterable[Any]) -> Dict[str, float]:
    model.train()
    loss_tracker = AverageMeter()

    for batch_idx, (X_batch, y_batch) in enumerate(train_gen, start=1):
        for sched in schedulers:
            sched.step()
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


def train_loop(dataset_dir_path: str, results_dir_path: str, architecture: str, force_cpu: bool = False,
               epochs: int = 100, batch_size: int = 256, shuffle: bool = True, base_feature_maps: int = 64,
               use_mixup=True, alpha: float = 1, val_split_size: float = 0.15, base_lr: float = 1e-3,
               max_lr: float = 1e-2, use_early_stopping: bool = True, early_stopping_patience: int = 15,
               optimizer: str = 'radam', use_lookahead: bool = True) -> None:
    architectures_lookup_table = {'resnet': Resnet1D}
    optimizers_lookup_table = {'adam': torch.optim.Adam, 'radam': RAdam}
    assert architecture in architectures_lookup_table, 'Specified model architecture unknown!'
    assert optimizer in optimizers_lookup_table, 'Specified optimizer unknown!'
    device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
    initialize_random_seeds(constants.RANDOM_SEED)

    # Create specified model.
    model = architectures_lookup_table[architecture](constants.DATASET_FEATURES_SHAPE[0],
                                                     base_feature_maps, constants.NB_DATASET_CLASSES).to(device)
    summary(model, input_size=constants.DATASET_FEATURES_SHAPE)  # Shape without including batch.

    # Load train/test data (includes validation set preparation).
    data = load_full_dataset(dataset_dir_path, create_val_subset=True, val_size=val_split_size,
                             random_seed=constants.RANDOM_SEED)
    train_dataset = PUTEEGGesturesDataset(data.X_train, data.y_train)
    val_dataset = PUTEEGGesturesDataset(data.X_val, data.y_val)
    test_dataset = PUTEEGGesturesDataset(data.X_test, data.y_test)
    train_gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_gen = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_gen = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)  # Note: this data is quite simple, no additional workers will be required for loading/processing.

    # Optimizer setup.
    base_opt = optimizers_lookup_table[optimizer](model.parameters(), lr=base_lr)
    optimizer = Lookahead(base_opt, k=5, alpha=0.5) if use_lookahead else base_opt

    # LR schedulers setup.
    epochs_per_half_clr_cycle = 4
    clr = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=len(train_gen) * epochs_per_half_clr_cycle,
                   mode='triangular2', cycle_momentum=False)
    schedulers = [clr]

    # Callbacks setup.
    callbacks = [
        ModelCheckpoint(results_dir_path, 'val_loss', {'model': model, 'optimizer': optimizer},
                        verbose=1, save_best_only=True),
    ]
    if use_early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min',
                                       patience=early_stopping_patience))  # Important: early stopping must be last on the list!

    # Training itself.
    loss_fnc = torch.nn.MultiLabelSoftMarginLoss(reduction='mean',
                                                 weight=torch.tensor(data.class_weights, dtype=torch.float32).to(device))
    metrics = []
    os.makedirs(results_dir_path, exist_ok=True)

    try:
        for ep in range(1, epochs + 1):
            epoch_stats = _epoch_train(model, train_gen, device, optimizer, loss_fnc, ep, use_mixup, alpha, schedulers)
            val_stats = _validate(model, loss_fnc, val_gen, device)
            epoch_stats.update(val_stats)

            print(f'\nEpoch {ep} train loss: {epoch_stats["loss"]:.4f}, '
                  f'val loss: {epoch_stats["val_loss"]:.5f}, val_acc: {epoch_stats["val_acc"]:.4f}')

            metrics.append(epoch_stats)
            for cb in callbacks:
                cb.on_epoch_end(ep, epoch_stats)
    except EarlyStoppingSignal as e:
        print(e)
        best_epoch = ep - callbacks[-1].patience
        model.load_state_dict(
            torch.load(osp.join(results_dir_path, f'ep_{best_epoch}_checkpoint.tar'))['model_state_dict'])

    # Check results on final test/holdout set:
    test_set_stats = _validate(model, loss_fnc, test_gen, device)
    print(f'\nFinal evaluation on test set: '
          f'test loss: {test_set_stats["val_loss"]:.5f}, test_acc: {test_set_stats["val_acc"]:.4f}')

    # Save metrics/last network/optimizer state
    save_json(osp.join(results_dir_path, 'test_set_stats.json'), test_set_stats)
    save_json(osp.join(results_dir_path, 'training_losses_and_metrics.json'), {'epochs_stats': metrics})
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': ep}, osp.join(results_dir_path, 'last_epoch_checkpoint.tar'))
