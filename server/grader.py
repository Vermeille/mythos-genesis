import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


class Permute(nn.Module):
    def __init__(self, *permute):
        super().__init__()
        self.dims = permute

    def forward(self, x):
        return x.permute(*self.dims)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.block(x)


def ConvGrader(dim, num_classes):
    return nn.Sequential(
        nn.Embedding(1025, dim),
        nn.LayerNorm(dim),
        Permute(0, 2, 1),
        Reshape(dim, 16, 16),
        ResBlock(dim),
        nn.ReLU(True),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim, num_classes, 1),
        Reshape(num_classes),
    )
