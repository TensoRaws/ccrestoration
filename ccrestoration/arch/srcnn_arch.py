import torch
from torch import nn
from torch.nn import functional as F

from ccrestoration.arch import ARCH_REGISTRY
from ccrestoration.type import ArchType
from ccrestoration.util.color import rgb_to_yuv, yuv_to_rgb


@ARCH_REGISTRY.register(name=ArchType.SRCNN)
class SRCNN(nn.Module):
    def __init__(self, num_channels: int = 1, scale: int = 2) -> None:
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.num_channels = num_channels
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear")

        if self.num_channels == 1 and x.size(1) == 3:
            # RGB -> YUV
            x = rgb_to_yuv(x)
            y, u, v = x[:, 0:1, ...], x[:, 1:2, ...], x[:, 2:3, ...]

            y = self.relu(self.conv1(y))
            y = self.relu(self.conv2(y))
            y = self.conv3(y)

            x = torch.cat([y, u, v], dim=1)
            # YUV -> RGB
            x = yuv_to_rgb(x)
        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)

        return x
