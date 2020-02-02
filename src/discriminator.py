from typing import Any

import torch
import torch.nn as nn


class Conv4x4LRelu(nn.Module):
    def __init__(self,
                 in_: int,
                 out: int,
                 negative_slop: float = 0.2
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_, out_channels=out, kernel_size=4, stride=2, padding=1, bias=False),
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.LeakyReLU(negative_slop, inplace=True)

        self.features = nn.Sequential(self.conv, self.bn, self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class Discriminator(nn.Module):
    def __init__(self,
                 channels: int = 3,
                 filters: int = 64,
                 gpu_devices: int = 1,
                 ):
        super().__init__()

        self.gpu_devices = gpu_devices

        self.features = nn.Sequential(
            Conv4x4LRelu(channels, filters),
            Conv4x4LRelu(filters, filters * 2),
            Conv4x4LRelu(filters * 4, filters * 8),
            nn.Conv2d(filters * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Any:
        if x.is_cuda and self.gpu_devices > 1:
            output = nn.parallel.data_parallel(self.features, x, range(self.gpu_devices))
        else:
            output = self.features(x)

        return output.view(-1, 1).squeeze(1)
