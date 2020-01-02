from typing import Tuple
import os.path as osp

import torch
from torch import nn
from torch.nn import functional as F
# from apex import amp  # Pytorch extension for automatical half/mixed precision optimizations
from torchsummary import summary  # another extension, a'la keras model.summary funtion


# TODO: move to utils
# relation for output size when using zero padding and non-unit strides: o = floor(i + 2p − k / s )+ 1
def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


class Triple1DConvResBlock(nn.Module):

    def __init__(self, input_channels: int, base_feature_maps: int):
        super(Triple1DConvResBlock, self).__init__()
        # Note: in pytorch the default ordering is channels first, so the input must be (N, C, L),
        # where N is batch size (omitted when creating model), L is sequence length and C is the number of features.
        # conv 7/8 kernel size padding same
        # batch norm
        # relu
        self.conv_1 = nn.Conv1d(input_channels, base_feature_maps, kernel_size=7,
                                stride=1, padding=_get_padding(7, 1, 1), bias=False)
        self.bn_1 = nn.BatchNorm1d(base_feature_maps)
        self.relu = nn.ReLU(inplace=True)  # TODO: use inplace?

        # conv 5 kernel size padding same
        # batch norm
        # relu
        self.conv_2 = nn.Conv1d(base_feature_maps, base_feature_maps, kernel_size=5,
                                stride=1, padding=_get_padding(5, 1, 1), bias=False)
        self.bn_2 = nn.BatchNorm1d(base_feature_maps)

        # conv 3 kernel size padding same
        # batch norm
        self.conv_3 = nn.Conv1d(base_feature_maps, base_feature_maps, kernel_size=3,
                                stride=1, padding=_get_padding(3, 1, 1), bias=False)
        self.bn_3 = nn.BatchNorm1d(base_feature_maps)

        # shortcut – conv 1x1, pad same o n the orig block input
        # batch norm
        self.skip_conv = nn.Conv1d(input_channels, base_feature_maps, kernel_size=1,
                                   stride=1, padding=_get_padding(1, 1, 1), bias=False)
        self.skip_bn = nn.BatchNorm1d(base_feature_maps)

        # add triple conv output + shortcut
        # relu

        # finito, return result

    def forward(self, x):
        # Block #1.
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        # Block #2.
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        # Block #3.
        out = self.conv_3(out)
        out = self.bn_3(out)

        # Skip connection
        skip_out = self.skip_conv(x)
        skip_out = self.skip_bn(skip_out)
        out += skip_out
        out = self.relu(out)

        return out


class GlobalAveragePooling1D(nn.Module):

    def forward(self, x):
        return torch.mean(x, -1)


class Resnet1D(nn.Module):

    def __init__(self, input_channels, base_feature_maps: int, nb_classes: int):
        super(Resnet1D, self).__init__()
        self.res_block_1 = Triple1DConvResBlock(input_channels, base_feature_maps)
        self.res_block_2 = Triple1DConvResBlock(base_feature_maps, base_feature_maps * 2)
        self.res_block_3 = Triple1DConvResBlock(base_feature_maps * 2, base_feature_maps * 2)
        self.gap = GlobalAveragePooling1D()
        self.dense = nn.Linear(base_feature_maps * 2, nb_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.res_block_1(x)
        out = self.res_block_2(out)
        out = self.res_block_3(out)
        out = self.gap(out)
        out = self.dense(out)
        out = self.softmax(out)

        return out



def train():
    epochs = 100
    batch_size = 256
    callbacks = None
    lr = 1e-3
    device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
    model = Resnet1D(input_shape[0], 64, 8).to(device)

    split_dir_path =  '/media/ja/CCTV_nagrania/mkm_archive/put_emg/data/raw_filtered_data_subjects_split_window_size_1024_window_stride_512_cv_splits_standarized/03/split_0'
    # X_train, y_train, X_test, y_test = None, None, None, None
    # TODO: X_val, y_val
    X_train = np.load(osp.join(split_dir_path, 'X_train.npy'))
    y_train = np.load(osp.join(split_dir_path, 'y_train.npy'))
    X_test = np.load(osp.join(split_dir_path, 'X_test.npy'))
    y_test = np.load(osp.join(split_dir_path, 'y_test.npy'))
    # swap axes as pytorch is channels first
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

    # add shuffle
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


if __name__ == '__main__':
    force_cpu = False
    # Note: in pytorch the default ordering is channels first, so the input must be (N, C, L),
    # where N is batch size (omitted when creating model), L is sequence length and C is the number of features.
    # Another note, on bias=False in conv layers. It should be in the model, and in fact is:
    # "And in pytorch the batchnorm implementation has weights and bias in addition to running mean and running standard deviation."
    # It is handled by batchnorm – actually it is not only pytorch specific:
    # Long story short: Even if you implement the ConvWithBias+BatchNorm, it will behave like ConvWithoutBias+BatchNorm.
    # It is the same as multiple fully-connected layers without activation function will behave like a single one.
    input_shape = (24, 1024)
    device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
    model = Resnet1D(input_shape[0], 64, 8).to(device)
    import numpy as np
    dummy_input = np.zeros((10, 24, 1024))
    dummy_input = torch.tensor(dummy_input, dtype=torch.float32).to(device)
    test_out = model(dummy_input)

    summary(model, input_size=input_shape)  # input shape is without batch
    # Overall, the total amount of weights is different than the keras model by roughly ~5k.
    # But it is probably due to redundant bias in  keras convolutions.

    train()
