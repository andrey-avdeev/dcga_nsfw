from typing import Any

import torch
import torch.nn as nn


class ConvTranspose4x4Relu(nn.Module):
    def __init__(self,
                 in_: int,
                 out: int):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_, out, kernel_size=4, stride=1, padding=0, bias=False),
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

        self.features = nn.Sequential(self.conv, self.bn, self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class Generator(nn.Module):
    def __init__(self,
                 z_size: int = 100,
                 channels: int = 3,
                 filters: int = 64,
                 gpu_devices: int = 1
                 ):
        super(Generator, self).__init__()
        self.gpu_devices = gpu_devices

        self.features = nn.Sequential(
            ConvTranspose4x4Relu(z_size, filters * 8),
            ConvTranspose4x4Relu(filters * 8, filters * 4),
            ConvTranspose4x4Relu(filters * 4, filters * 2),
            ConvTranspose4x4Relu(filters * 2, filters),
            nn.ConvTranspose2d(filters, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Any:
        if x.is_cuda and self.gpu_devices > 1:
            output = nn.parallel.data_parallel(self.features, x, range(self.gpu_devices))
        else:
            output = self.features(x)

        return output
