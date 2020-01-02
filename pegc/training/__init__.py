import torch
from torch import nn
from torchsummary import summary  # another extension, a'la keras model.summary funtion

from pegc.models import Resnet1D
from pegc import constants
from pegc.training.utils import load_full_dataset


def train_loop(dataset_dir_path: str, architecture: str, force_cpu: bool = False, epochs: int = 100,
               batch_size: int = 256, base_feature_maps: int = 64):
    architectures_lookup_table = {'resnet': Resnet1D}
    assert architecture in architectures_lookup_table, 'Specified model architecture unknown!'
    device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')

    model = architectures_lookup_table[architecture](constants.DATASET_FEATURES_SHAPE[0],
                                                     base_feature_maps, constants.NB_DATASET_CLASSES).to(device)
    summary(model, input_size=constants.DATASET_FEATURES_SHAPE)  # Without including batch.

    X_train, y_train, X_test, y_test = load_full_dataset(dataset_dir_path)
    # TODO: X_val, y_val

    callbacks = None  # TODO

    # TODO: also make adjustable? Perhaps some config file would be more handy?
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

    # TODO: add shuffle
    # TODO/maybe: some augmentation, mixup maybe?
    for ep in range(epochs):
        model.train()

        loss_sum = 0
        acc_sum = 0
        nb_batches = 0

        for b in range(0, len(X_train), batch_size):
            X_batch = torch.tensor(X_train[b: b + batch_size], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_train[b: b + batch_size], dtype=torch.float32).to(device)

            batch_pred = model(X_batch)
            loss = loss_criterion(batch_pred, y_batch)

            loss_sum += loss
            acc_sum += (y_batch.argmax(dim=1) == batch_pred.argmax(dim=1)).sum()
            nb_batches += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch {ep} loss: {loss_sum.item() / len(X_train):.5f} acc: {acc_sum.item() / len(X_train):.5f}')
