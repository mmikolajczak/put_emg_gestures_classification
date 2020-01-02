from pegc.models.resnet_1d import Resnet1D


# General notes for all pytorch models:
# Note: in pytorch the default ordering is channels first, so the input must be (N, C, L),
# where N is batch size (omitted when creating model), L is sequence length and C is the number of features.
# Another note, on bias=False in conv layers. It should be in the model, and in fact is:
# "And in pytorch the batchnorm implementation has weights and bias in addition to running mean and running standard deviation."
# It is handled by batchnorm – actually it is not only pytorch specific:
# Long story short: Even if you implement the ConvWithBias+BatchNorm, it will behave like ConvWithoutBias+BatchNorm.
# It is the same as multiple fully-connected layers without activation function will behave like a single one.
# (Meaning keras is a bit redundant in this matter).
