import torch
from torch import nn


# Relation for conv output size when using zero padding and non-unit strides: o = floor(i + 2p − k / s )+ 1.
def _get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class GlobalAveragePooling1D(nn.Module):

    def forward(self, x):
        return torch.mean(x, -1)
