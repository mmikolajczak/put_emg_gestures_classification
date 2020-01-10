from torch import nn

from pegc.models.utils import _get_padding, GlobalAveragePooling1D


class Triple1DConvResBlock(nn.Module):

    def __init__(self, input_channels: int, base_feature_maps: int):
        super(Triple1DConvResBlock, self).__init__()
        # Note: in pytorch the default ordering is channels first, so the input must be (N, C, L),
        # where N is batch size (omitted when creating model), L is sequence length and C is the number of features.
        # Block #1.
        self.conv_1 = nn.Conv1d(input_channels, base_feature_maps, kernel_size=7,
                                stride=1, padding=_get_padding(7, 1, 1), bias=False)
        self.bn_1 = nn.BatchNorm1d(base_feature_maps)
        self.relu = nn.ReLU(inplace=True)  # TODO: use inplace?

        # Block #2.
        self.conv_2 = nn.Conv1d(base_feature_maps, base_feature_maps, kernel_size=5,
                                stride=1, padding=_get_padding(5, 1, 1), bias=False)
        self.bn_2 = nn.BatchNorm1d(base_feature_maps)
        # Block #3.
        self.conv_3 = nn.Conv1d(base_feature_maps, base_feature_maps, kernel_size=3,
                                stride=1, padding=_get_padding(3, 1, 1), bias=False)
        self.bn_3 = nn.BatchNorm1d(base_feature_maps)

        # Skip connectipon
        self.skip_conv = nn.Conv1d(input_channels, base_feature_maps, kernel_size=1,
                                   stride=1, padding=_get_padding(1, 1, 1), bias=False)
        self.skip_bn = nn.BatchNorm1d(base_feature_maps)

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


class Resnet1D(nn.Module):

    def __init__(self, input_channels, nb_classes: int, nb_res_blocks: int, res_blocks_per_expansion: int,
                 base_feature_maps: int):
        super(Resnet1D, self).__init__()
        res_blocks = []
        res_input_shape = input_channels
        nb_out_feature_maps = base_feature_maps
        for i in range(nb_res_blocks):
            if (i + 1) % res_blocks_per_expansion == 0 and not i == nb_res_blocks - 1:
                nb_out_feature_maps = nb_out_feature_maps * 2
            cur_block = Triple1DConvResBlock(res_input_shape, nb_out_feature_maps)
            res_blocks.append(cur_block)
            res_input_shape = nb_out_feature_maps
        self.res_blocks = nn.Sequential(*res_blocks)
        self.gap = GlobalAveragePooling1D()
        self.dense = nn.Linear(nb_out_feature_maps, nb_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = x
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.gap(out)
        out = self.dense(out)
        out = self.softmax(out)

        return out
